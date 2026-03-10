"""
LSTM Price Prediction Training
"""
import os
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

from feature_engineering import prepare_features, select_features
from common.data_utils import load_data, create_sequences, detect_symbol
from common.id_utils import get_next_model_id, extract_data_id
from common.visualization import plot_training_curves

warnings.filterwarnings('ignore')

# All feature set names available in universal FE_
_ALL_FEATURE_SETS = [
    # Universal
    'full', 'technical', 'technical_bp', 'minimal',
    # LSTM-dedicated
    'lstm_momentum', 'lstm_volatility', 'lstm_price',
    # XGBoost-dedicated (available but not recommended for LSTM)
    'xgb_momentum', 'xgb_volatility', 'xgb_returns',
    # Placeholders
    'k_set1', 'k_set2', 'k_set3',
]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPriceModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super(LSTMPriceModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def setup_logger(run_dir):
    logger = logging.getLogger('price_training')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(run_dir, 'training.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_scheduler(optimizer, scheduler_type, **kwargs):
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=kwargs.get('t0', 10), T_mult=kwargs.get('mult', 2))
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=kwargs.get('patience', 5),
            factor=kwargs.get('factor', 0.5), mode='min')
    return None


def _read_use_log_prices(metadata):
    """
    Read use_log_prices from metadata — handles both formats:
      - Universal (new):  metadata['scalers']['use_log_prices']
      - Legacy (old):     metadata['use_log_prices']
    Defaults to True when key is absent.
    """
    # New universal format
    scalers = metadata.get('scalers')
    if isinstance(scalers, dict) and 'use_log_prices' in scalers:
        return scalers['use_log_prices']
    # Legacy format
    if 'use_log_prices' in metadata:
        return metadata['use_log_prices']
    return True


def _train_single_model(
        train_path, lookback, horizon, hidden_size, num_layers, dropout,
        epochs, batch_size, lr, weight_decay, use_subset, subset_size,
        feature_set, custom_features, validate_features_flag, generate_features,
        scheduler_type, scheduler_params, use_log_prices, early_stopping_patience,
        seed, optuna_config_name=None
):
    data, is_prepared, metadata = load_data(train_path, use_subset, subset_size)

    if is_prepared:
        symbol = metadata.get('symbol', 'crypto')
        data_id = extract_data_id(metadata)
    else:
        filename = os.path.basename(train_path)
        symbol = detect_symbol(filename)
        data_id = 'UNKNOWN'

    model_id = get_next_model_id()
    date_short = datetime.now().strftime('%d%m')

    if optuna_config_name:
        run_name = f"{model_id}_{symbol}_model_{data_id}_{optuna_config_name}_{date_short}"
    else:
        run_name = f"{model_id}_{symbol}_model_{data_id}_{date_short}"

    base_dir = 'lstm_models'
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(run_dir)

    logger.info("=" * 80)
    logger.info(f"LSTM PRICE PREDICTION - {symbol.upper()} - Horizon t+{horizon}")
    logger.info("=" * 80)
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Data ID:  {data_id}")
    if optuna_config_name:
        logger.info(f"Optuna:   {optuna_config_name}")
    logger.info(f"Output:   {run_dir}")
    logger.info(f"Seed:     {seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device:   {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU:      {torch.cuda.get_device_name(0)}")

    logger.info(f"\nLoading data from {train_path}...")

    if is_prepared:
        logger.info("Mode: PREPARED DATA (NPY)")

        # --- Metadata keys: handle universal vs legacy format ---
        # Universal: metadata has 'symbol', no 'source_file'
        # Legacy:    metadata has 'source_file'
        source_label = metadata.get('source_file') or metadata.get('symbol', 'unknown').upper()
        logger.info(f"Source: {source_label}")

        X_train = data['X_train']
        y_train = data['y_train']
        X_val   = data['X_val']
        y_val   = data['y_val']
        feature_scaler = data['feature_scaler']
        price_scaler   = data['price_scaler']

        # Read params locked in the prepared data
        features       = metadata['features']
        lookback       = metadata['lookback']
        horizon        = metadata['horizon']
        use_log_prices = _read_use_log_prices(metadata)
        has_buy_pressure = metadata.get('has_buy_pressure',
                                        'buy_pressure' in features)

        logger.info(f"Train:        {len(X_train):,} sequences  {X_train.shape}")
        logger.info(f"Val (test):   {len(X_val):,} sequences  {X_val.shape}")
        logger.info(f"Features:     {len(features)}")
        logger.info(f"Lookback:     {lookback}")
        logger.info(f"Horizon:      t+{horizon}")
        logger.info(f"Log prices:   {use_log_prices}")
        logger.info(f"buy_pressure: {'included' if has_buy_pressure else 'excluded'}")

    else:
        logger.info("Mode: RAW DATA (CSV)")
        df = data
        logger.info(f"Loaded {len(df)} records")

        if generate_features:
            df = prepare_features(df)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        logger.info(f"Dataset size: {len(df)}")

        # Auto-detect buy_pressure availability
        has_buy_pressure = 'buy_pressure' in df.columns
        if has_buy_pressure:
            logger.info("buy_pressure: included (taker_buy column detected)")
        else:
            logger.info("buy_pressure: excluded (no taker_buy column)")

        features = select_features(
            df,
            feature_set=feature_set,
            custom_features=custom_features,
            validate=validate_features_flag,
            include_buy_pressure=has_buy_pressure,
        )
        features = [f for f in features if f != 'close']
        logger.info(f"Features: {len(features)}")

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        val_df   = df.iloc[split_idx:].copy()
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

        feature_scaler = StandardScaler()
        train_df[features] = feature_scaler.fit_transform(train_df[features])
        val_df[features]   = feature_scaler.transform(val_df[features])

        if use_log_prices:
            train_df['close'] = np.log(train_df['close'])
            val_df['close']   = np.log(val_df['close'])

        price_scaler = StandardScaler()
        train_df['close'] = price_scaler.fit_transform(
            train_df['close'].values.reshape(-1, 1)).flatten()
        val_df['close'] = price_scaler.transform(
            val_df['close'].values.reshape(-1, 1)).flatten()

        X_train, y_train, _ = create_sequences(train_df, features, lookback, horizon)
        X_val,   y_val,   _ = create_sequences(val_df,   features, lookback, horizon)

    # ------------------------------------------------------------------ #
    # DataLoaders
    # ------------------------------------------------------------------ #
    train_loader = DataLoader(
        CryptoDataset(X_train, y_train), batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=0,
        pin_memory=device.type == 'cuda',
    )
    val_loader = DataLoader(
        CryptoDataset(X_val, y_val), batch_size=batch_size,
        shuffle=False, num_workers=0,
        pin_memory=device.type == 'cuda',
    )

    model = LSTMPriceModel(len(features), hidden_size, num_layers, dropout).to(device)

    logger.info(f"\nModel: Hidden={hidden_size}, Layers={num_layers}, Dropout={dropout}")
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, scheduler_type, **scheduler_params)

    logger.info(f"Training: Epochs={epochs}, Batch={batch_size}, LR={lr}")

    best_val_loss = float('inf')
    patience = 0
    train_losses = []
    val_losses   = []

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING START")
    logger.info("=" * 80)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(run_dir, 'lstm_price_model.pth'))
            logger.info(f"  >> NEW BEST")
        else:
            patience += 1
            if patience >= early_stopping_patience:
                logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                break

    logger.info("\nSaving model package...")

    joblib.dump(feature_scaler, os.path.join(run_dir, 'feature_scaler.pkl'))
    joblib.dump(price_scaler,   os.path.join(run_dir, 'price_scaler.pkl'))

    config = {
        'model_id':            model_id,
        'data_id':             data_id,
        'features':            features,
        'lookback':            lookback,
        'horizon':             horizon,
        'hidden_size':         hidden_size,
        'num_layers':          num_layers,
        'dropout':             dropout,
        'feature_set':         feature_set,
        'custom_features':     custom_features,
        'validate_features':   validate_features_flag,
        'generate_features':   generate_features,
        'use_log_prices':      use_log_prices,
        'has_buy_pressure':    has_buy_pressure,
        'model_type':          'price_prediction',
        'symbol':              symbol,
        'timestamp':           datetime.now().strftime('%Y%m%d_%H%M%S'),
        'best_val_loss':       best_val_loss,
        'train_path':          train_path,
        'scheduler_type':      scheduler_type,
        'data_source':         'prepared_npy' if is_prepared else 'raw_csv',
        'prepared_metadata':   metadata if is_prepared else None,
        'optuna_config':       optuna_config_name,
        'output_dir':          run_dir,
    }
    joblib.dump(config, os.path.join(run_dir, 'model_config.pkl'))

    plot_training_curves(train_losses, val_losses, best_val_loss,
                         os.path.join(run_dir, 'training_curves.png'))

    logger.info(f"Model ID: {model_id}, Best val loss: {best_val_loss:.6f}")
    logger.info(f"Output: {run_dir}")

    return model, feature_scaler, price_scaler, config


def train_model(
        train_path='lstm_prepared_data',
        lookback=48, horizon=1, hidden_size=128, num_layers=1, dropout=0.2,
        epochs=100, batch_size=256, lr=0.0005, weight_decay=1e-5,
        use_subset=False, subset_size=100000,
        feature_set=None, custom_features=None,
        validate_features_flag=True, generate_features=True,
        scheduler_type='none', scheduler_params=None,
        use_log_prices=True, early_stopping_patience=25,
        load_optuna_params=None, load_optuna_dir=None, train_all_top=False,
        seed=42
):
    """Train LSTM for cryptocurrency price prediction."""
    set_seed(seed)

    if scheduler_params is None:
        scheduler_params = {}

    optuna_configs = []

    if load_optuna_params is not None:
        with open(load_optuna_params, 'r') as f:
            optuna_params = json.load(f)
        metadata = optuna_params.pop('_metadata', None)
        optuna_configs.append({'params': optuna_params, 'metadata': metadata, 'name': 'single'})

    elif load_optuna_dir is not None:
        if os.path.sep not in load_optuna_dir and not os.path.exists(load_optuna_dir):
            load_optuna_dir = os.path.join('lstm_data', load_optuna_dir)
        if not os.path.exists(load_optuna_dir):
            raise FileNotFoundError(f"Optuna directory not found: {load_optuna_dir}")

        param_files = (['best_params.json', 'second_best_params.json', 'third_best_params.json']
                       if train_all_top else ['best_params.json'])
        names = ['best', 'second_best', 'third_best']

        for param_file, name in zip(param_files, names):
            param_path = os.path.join(load_optuna_dir, param_file)
            if os.path.exists(param_path):
                with open(param_path, 'r') as f:
                    optuna_params = json.load(f)
                metadata = optuna_params.pop('_metadata', None)
                optuna_configs.append({'params': optuna_params, 'metadata': metadata, 'name': name})

    if len(optuna_configs) > 1:
        results = []
        for idx, cfg in enumerate(optuna_configs):
            print(f"\n{'#' * 80}")
            print(f"# TRAINING MODEL {idx + 1}/{len(optuna_configs)}: {cfg['name'].upper()}")
            print(f"{'#' * 80}\n")
            op = cfg['params']
            md = cfg['metadata']
            result = _train_single_model(
                train_path=train_path,
                lookback=op.get('lookback', lookback),
                horizon=op.get('horizon', horizon),
                hidden_size=op.get('hidden_size', hidden_size),
                num_layers=op.get('num_layers', num_layers),
                dropout=op.get('dropout', dropout),
                epochs=epochs,
                batch_size=op.get('batch_size', batch_size),
                lr=op.get('learning_rate', lr),
                weight_decay=op.get('weight_decay', weight_decay),
                use_subset=use_subset,
                subset_size=subset_size,
                feature_set=md.get('feature_set') if md else feature_set,
                custom_features=md.get('custom_features') if md else custom_features,
                validate_features_flag=validate_features_flag,
                generate_features=generate_features,
                scheduler_type=op.get('scheduler_type', scheduler_type),
                scheduler_params=scheduler_params,
                use_log_prices=use_log_prices,
                early_stopping_patience=early_stopping_patience,
                seed=seed,
                optuna_config_name=cfg['name'],
            )
            results.append(result)

        print(f"\n{'#' * 80}")
        print(f"# TRAINED {len(results)} MODELS SUCCESSFULLY")
        print(f"{'#' * 80}\n")
        return results

    if optuna_configs:
        cfg = optuna_configs[0]
        op  = cfg['params']
        md  = cfg['metadata']
        lookback      = op.get('lookback', lookback)
        horizon       = op.get('horizon', horizon)
        hidden_size   = op.get('hidden_size', hidden_size)
        num_layers    = op.get('num_layers', num_layers)
        dropout       = op.get('dropout', dropout)
        batch_size    = op.get('batch_size', batch_size)
        lr            = op.get('learning_rate', lr)
        weight_decay  = op.get('weight_decay', weight_decay)
        scheduler_type = op.get('scheduler_type', scheduler_type)
        if md:
            feature_set     = md.get('feature_set', feature_set)
            custom_features = md.get('custom_features', custom_features)

    return _train_single_model(
        train_path, lookback, horizon, hidden_size, num_layers, dropout,
        epochs, batch_size, lr, weight_decay, use_subset, subset_size,
        feature_set, custom_features, validate_features_flag, generate_features,
        scheduler_type, scheduler_params, use_log_prices, early_stopping_patience,
        seed, None
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='lstm_prepared_data')
    parser.add_argument('--lookback', type=int, default=48)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--use_subset', action='store_true')
    parser.add_argument('--subset_size', type=int, default=100000)
    parser.add_argument('--feature_set', type=str, default=None, choices=_ALL_FEATURE_SETS)
    parser.add_argument('--custom_features', type=str, nargs='+', default=None)
    parser.add_argument('--no_validate_features', action='store_true')
    parser.add_argument('--no_generate_features', action='store_true')
    parser.add_argument('--scheduler_type', type=str, default='none')
    parser.add_argument('--use_log_prices', action='store_true', default=True)
    parser.add_argument('--early_stopping_patience', type=int, default=25)
    parser.add_argument('--load_optuna_params', type=str, default=None)
    parser.add_argument('--load_optuna_dir', type=str, default=None)
    parser.add_argument('--train_all_top', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train_model(
        train_path=args.train_path,
        lookback=args.lookback,
        horizon=args.horizon,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_subset=args.use_subset,
        subset_size=args.subset_size,
        feature_set=args.feature_set,
        custom_features=args.custom_features,
        validate_features_flag=not args.no_validate_features,
        generate_features=not args.no_generate_features,
        scheduler_type=args.scheduler_type,
        use_log_prices=args.use_log_prices,
        early_stopping_patience=args.early_stopping_patience,
        load_optuna_params=args.load_optuna_params,
        load_optuna_dir=args.load_optuna_dir,
        train_all_top=args.train_all_top,
        seed=args.seed,
    )
