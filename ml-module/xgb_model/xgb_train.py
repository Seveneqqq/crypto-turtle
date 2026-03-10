"""
XGBoost Price Prediction Training
"""
import os
import warnings
from datetime import datetime

import numpy as np

from xgb_config import XGBConfig
from common.data_utils import load_data
from common.id_utils import get_next_model_id, extract_data_id, detect_symbol
from common.training_utils import (
    set_seed, setup_logger,
    train_model as _train_model,
    save_xgb_model, print_final_summary
)

warnings.filterwarnings('ignore')


def train(
        data_path,
        lookback=None,
        horizon=None,
        flattening_method=None,
        feature_set=None,
        custom_features=None,
        n_estimators=None,
        max_depth=None,
        learning_rate=None,
        subsample=None,
        colsample_bytree=None,
        colsample_bylevel=None,
        min_child_weight=None,
        gamma=None,
        reg_alpha=None,
        reg_lambda=None,
        objective=None,
        early_stopping_rounds=None,
        seed=42,
        verbose=True,
        scale_target=None,
        target_scaler_type=None,
        winsorize_target=None,
        winsorize_percentile=None,
        load_optuna_params=None,
):
    set_seed(seed)

    # Load Optuna params — override only values not explicitly provided
    if load_optuna_params is not None:
        optuna_params, optuna_meta = XGBConfig.load_optuna_params(load_optuna_params)
        print(f"Loaded Optuna params from: {load_optuna_params}")

        if lookback is None:           lookback           = optuna_params.get('lookback')
        if horizon is None:            horizon            = optuna_params.get('horizon')
        if flattening_method is None:  flattening_method  = optuna_params.get('flattening_method')
        if n_estimators is None:       n_estimators       = optuna_params.get('n_estimators')
        if max_depth is None:          max_depth          = optuna_params.get('max_depth')
        if learning_rate is None:      learning_rate      = optuna_params.get('learning_rate')
        if subsample is None:          subsample          = optuna_params.get('subsample')
        if colsample_bytree is None:   colsample_bytree   = optuna_params.get('colsample_bytree')
        if colsample_bylevel is None:  colsample_bylevel  = optuna_params.get('colsample_bylevel')
        if min_child_weight is None:   min_child_weight   = optuna_params.get('min_child_weight')
        if gamma is None:              gamma              = optuna_params.get('gamma')
        if reg_alpha is None:          reg_alpha          = optuna_params.get('reg_alpha')
        if reg_lambda is None:         reg_lambda         = optuna_params.get('reg_lambda')
        if objective is None:          objective          = optuna_params.get('objective')
        if scale_target is None:       scale_target       = optuna_meta.get('scale_target')
        if target_scaler_type is None: target_scaler_type = optuna_meta.get('target_scaler_type')
        if feature_set is None:        feature_set        = optuna_meta.get('feature_set')

    # Apply config defaults
    if lookback is None:              lookback              = XGBConfig.DEFAULT_LOOKBACK
    if horizon is None:               horizon               = XGBConfig.DEFAULT_HORIZON
    if flattening_method is None:     flattening_method     = XGBConfig.DEFAULT_FLATTENING
    if feature_set is None:           feature_set           = XGBConfig.DEFAULT_FEATURE_SET
    if early_stopping_rounds is None: early_stopping_rounds = XGBConfig.EARLY_STOPPING_ROUNDS
    if scale_target is None:          scale_target          = XGBConfig.SCALE_TARGET
    if target_scaler_type is None:    target_scaler_type    = XGBConfig.TARGET_SCALER_TYPE
    if winsorize_target is None:      winsorize_target      = XGBConfig.WINSORIZE_TARGET
    if winsorize_percentile is None:  winsorize_percentile  = XGBConfig.WINSORIZE_PERCENTILE

    # Build XGBoost params
    xgb_objective, extra_params = XGBConfig.get_xgb_objective()
    params = {**XGBConfig.DEFAULT_PARAMS, 'objective': xgb_objective, **extra_params}

    overrides = {
        'n_estimators': n_estimators, 'max_depth': max_depth,
        'learning_rate': learning_rate, 'subsample': subsample,
        'colsample_bytree': colsample_bytree, 'colsample_bylevel': colsample_bylevel,
        'min_child_weight': min_child_weight, 'gamma': gamma,
        'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda,
    }
    # objective override from Optuna/CLI (already a real XGBoost string, no large-delta handling)
    INCOMPATIBLE_OBJECTIVES = {'reg:squaredlogerror', 'rank:ndcg'}
    if objective is not None:
        if objective in INCOMPATIBLE_OBJECTIVES:
            raise ValueError(
                f"Objective '{objective}' is incompatible with log-return targets.\n"
                f"  reg:squaredlogerror — requires label > -1 (scaled log-returns can be negative)\n"
                f"  rank:ndcg          — requires non-negative integer labels\n"
                f"Use one of: reg:pseudohubererror, reg:squarederror, reg:absoluteerror"
            )
        params['objective'] = objective
    for k, v in overrides.items():
        if v is not None:
            params[k] = v

    print("=" * 80)
    print("XGBOOST TRAINING")
    print("=" * 80)

    symbol = detect_symbol(os.path.basename(data_path))

    data, _, metadata = load_data(
        data_path,
        flattening_method=flattening_method,
        feature_set=feature_set,
        custom_features=custom_features,
        lookback=lookback,
        horizon=horizon,
        scale_target=scale_target,
        target_scaler_type=target_scaler_type,
        winsorize_target=winsorize_target,
        winsorize_percentile=winsorize_percentile,
    )

    data_id        = extract_data_id(metadata)
    features       = metadata['features']
    flat_features  = metadata['flat_features']
    lookback       = metadata['lookback']
    horizon        = metadata['horizon']
    flattening_method = metadata['flattening_method']
    clip_bounds    = data.get('clip_bounds')

    X_train        = data['X_train']
    y_train        = data['y_train']
    X_val          = data['X_val']
    y_val          = data['y_val']
    feature_scaler = data['feature_scaler']
    target_scaler  = data.get('target_scaler')

    print(f"\nData ID:    {data_id}")
    print(f"Features:   {len(features)} → {len(flat_features)} flat")
    print(f"Flattening: {flattening_method}")
    print(f"Lookback:   {lookback}, Horizon: {horizon}")
    print(f"Train:      {len(X_train)} samples, Val: {len(X_val)} samples")
    print(f"Target scaler:   {'ENABLED (' + target_scaler.method + ')' if target_scaler else 'DISABLED'}")
    print(f"Winsorisation:   {'ENABLED (' + str(winsorize_percentile) + 'th pct)' if winsorize_target else 'DISABLED'}")

    model_id   = get_next_model_id()
    date_short = datetime.now().strftime('%d%m')
    run_name   = f"{model_id}_{symbol}_xgb_model_{data_id}_{date_short}"
    output_dir = os.path.join('xgb_models', run_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info("=" * 80)
    logger.info(f"XGBOOST TRAINING - {symbol.upper()} - Horizon t+{horizon}")
    logger.info("=" * 80)
    logger.info(f"Model ID:   {model_id}")
    logger.info(f"Data ID:    {data_id}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Target scaler:   {'ENABLED (' + target_scaler.method + ')' if target_scaler else 'DISABLED'}")
    logger.info(f"Winsorisation:   {'ENABLED (' + str(winsorize_percentile) + 'th pct)' if winsorize_target else 'DISABLED'}")
    if clip_bounds:
        logger.info(f"Clip bounds:     [{clip_bounds['lower']:.6f}, {clip_bounds['upper']:.6f}]")
    logger.info("\nHyperparameters:")
    for key, value in sorted(params.items()):
        logger.info(f"  {key:20s}: {value}")
    logger.info(f"  early_stopping:      {early_stopping_rounds}")

    use_best_iteration = XGBConfig.USE_BEST_ITERATION
    logger.info(f"Save from:  {'best_iteration' if use_best_iteration else 'last_iteration'}")

    model, best_iteration, best_score, train_metrics, val_metrics = _train_model(
        X_train, y_train, X_val, y_val,
        params, early_stopping_rounds,
        logger, verbose,
        target_scaler=target_scaler,
        use_best_iteration=use_best_iteration,
    )

    logger.info("\nTRAIN METRICS:")
    logger.info(f"  MAE:      {train_metrics['mae']:.6f}")
    logger.info(f"  RMSE:     {train_metrics['rmse']:.6f}")
    logger.info(f"  R²:       {train_metrics['r2']:.4f}")
    logger.info(f"  Dir Acc:  {train_metrics['dir_acc']:.2f}%")
    logger.info("\nVAL METRICS:")
    logger.info(f"  MAE:      {val_metrics['mae']:.6f}")
    logger.info(f"  RMSE:     {val_metrics['rmse']:.6f}")
    logger.info(f"  R²:       {val_metrics['r2']:.4f}")
    logger.info(f"  Dir Acc:  {val_metrics['dir_acc']:.2f}%")

    config = {
        'model_id':          model_id,
        'data_id':           data_id,
        'symbol':            symbol,
        'model_type':        'xgboost',
        'target_type':       'log_return',
        'features':          features,
        'flat_features':     flat_features,
        'n_features':        len(features),
        'n_flat_features':   len(flat_features),
        'lookback':          lookback,
        'horizon':           horizon,
        'flattening_method': flattening_method,
        'feature_set':       feature_set,
        'params':            params,
        'best_iteration':    int(best_iteration),
        'best_score':        float(best_score),
        'train_metrics':     train_metrics,
        'val_metrics':       val_metrics,
        'timestamp':         datetime.now().strftime('%Y%m%d_%H%M%S'),
        'output_dir':        output_dir,
        'early_stopping_rounds': early_stopping_rounds,
        'scale_target':      scale_target,
        'target_scaler_type': target_scaler_type if scale_target else None,
        'winsorize_target':  winsorize_target,
        'winsorize_percentile': winsorize_percentile if winsorize_target else None,
        'clip_bounds':       clip_bounds,
        'train_path':        data_path,
    }

    save_xgb_model(model, feature_scaler, config, output_dir, logger, target_scaler=target_scaler)
    print_final_summary(model_id, output_dir, train_metrics, val_metrics, logger)

    return model, feature_scaler, config


# Keep backward-compatible alias
train_model = train


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--load_optuna_params', type=str, default=None,
                        help='Path to Optuna best_params.json')
    parser.add_argument('--lookback', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--flattening_method', type=str, default=None,
                        choices=['lag', 'agg', 'hybrid'])
    parser.add_argument('--feature_set', type=str, default=None)
    parser.add_argument('--n_estimators', type=int, default=None)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--subsample', type=float, default=None)
    parser.add_argument('--colsample_bytree', type=float, default=None)
    parser.add_argument('--colsample_bylevel', type=float, default=None)
    parser.add_argument('--min_child_weight', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--reg_alpha', type=float, default=None)
    parser.add_argument('--reg_lambda', type=float, default=None)
    parser.add_argument('--objective', type=str, default=None,
                        choices=['reg:pseudohubererror', 'reg:pseudohubererror_large',
                                 'reg:squarederror', 'reg:absoluteerror'])
    parser.add_argument('--early_stopping_rounds', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale_target', action='store_true', default=None)
    parser.add_argument('--no_scale_target', dest='scale_target', action='store_false')
    parser.add_argument('--target_scaler_type', type=str, default=None,
                        choices=['standard', 'robust'])
    parser.add_argument('--winsorize_target', action='store_true', default=None,
                        help='Enable target winsorisation (clips extreme log-returns)')
    parser.add_argument('--winsorize_percentile', type=float, default=None,
                        help='Winsorisation percentile, e.g. 99.0 (clips outer 1%%)')

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        load_optuna_params=args.load_optuna_params,
        lookback=args.lookback,
        horizon=args.horizon,
        flattening_method=args.flattening_method,
        feature_set=args.feature_set,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        colsample_bylevel=args.colsample_bylevel,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        objective=args.objective,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        scale_target=args.scale_target,
        target_scaler_type=args.target_scaler_type,
        winsorize_target=args.winsorize_target,
        winsorize_percentile=args.winsorize_percentile,
    )
