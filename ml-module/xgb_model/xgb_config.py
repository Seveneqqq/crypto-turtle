"""
XGBoost Model Configuration and Optuna Integration
"""
import os
import json
import argparse


class XGBConfig:
    """Configuration for XGBoost model"""

    # DATA PREPARATION
    DEFAULT_LOOKBACK = 48
    DEFAULT_HORIZON = 12
    DEFAULT_FLATTENING = 'agg'
    DEFAULT_FEATURE_SET = 'full'

    # TARGET SCALING
    SCALE_TARGET = True
    TARGET_SCALER_TYPE = 'robust'               # 'standard' | 'robust'

    # TARGET WINSORISATION
    # Clips extreme log-return targets to [p, 1-p] percentile range computed
    # on the TRAINING set only. Val/test are clipped with the same bounds
    # (no leakage). Set WINSORIZE_TARGET = False to disable entirely.
    #
    # Recommended: 99.0 (clips top/bottom 0.5% — flash crashes, pump&dumps)
    # Conservative: 99.5 (clips top/bottom 0.25%)
    # Disabled:     set WINSORIZE_TARGET = False
    WINSORIZE_TARGET = False
    WINSORIZE_PERCENTILE = 99.0                 # e.g. 99.0 → clips at [0.5%, 99.5%]

    # OPTUNA OPTIMIZATION TARGET
    # Controls which dataset Optuna uses to compute the loss it minimises.
    #   'val'   - minimise loss on validation set (default, recommended)
    #             → selects params that generalise best
    #   'train' - minimise loss on training set
    #             → useful to diagnose underfitting, not recommended for production
    # The loss function used is always the one defined by OBJECTIVE above
    # (e.g. Pseudo-Huber), NOT sklearn MAE.
    OPTUNA_OPTIMIZE_TARGET = 'train'          # 'val' | 'train'

    # OBJECTIVE FUNCTION
    # Compatible options (log-return targets, values can be negative):
    #   'reg:pseudohubererror'          - Pseudo-Huber (default, robust to outliers)
    #   'reg:pseudohubererror_large'    - Pseudo-Huber with large delta (huber_slope=10, closer to MSE)
    #   'reg:absoluteerror'             - MAE (median regression, prone to flattening) 0
    #   'reg:squarederror'              - MSE (mean regression, sensitive to outliers)
    #
    # INCOMPATIBLE — do not use with log-return targets:
    #   'reg:squaredlogerror'           - requires label > -1 (fails with scaled log-returns)
    #   'rank:ndcg'                     - requires non-negative integer labels
    OBJECTIVE = 'reg:pseudohubererror_large'

    # Huber delta — only used when OBJECTIVE == 'reg:pseudohubererror_large'
    # Larger value = more MSE-like behaviour, better at capturing extremes
    HUBER_SLOPE = 10.0

    @classmethod
    def get_xgb_objective(cls):
        """
        Returns (objective_str, extra_params_dict).
        'reg:pseudohubererror_large' is a virtual name handled here —
        XGBoost receives 'reg:pseudohubererror' + huber_slope kwarg.
        """
        if cls.OBJECTIVE == 'reg:pseudohubererror_large':
            return 'reg:pseudohubererror', {'huber_slope': cls.HUBER_SLOPE}
        return cls.OBJECTIVE, {}

    # XGBOOST HYPERPARAMETERS
    DEFAULT_PARAMS = {
        'n_estimators': 2222,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'min_child_weight': 3,
        'gamma': 0.0,
        'reg_alpha': 0.0,
        'reg_lambda': 1,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
    }

    # TRAINING
    EARLY_STOPPING_ROUNDS = 20000
    VERBOSE_EVAL = 60
    # USE_BEST_ITERATION:
    #   False (default) - save model from last iteration (all n_estimators trees)
    #   True            - save model from best_iteration (early stopping checkpoint)
    USE_BEST_ITERATION = False

    # PATHS
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PREPARED_DATA_DIR = os.path.join(BASE_DIR, 'prepared_data')
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'xgb_models')
    LOGS_DIR = os.path.join(BASE_DIR, 'xgb_data')

    # ID PREFIXES
    DATA_ID_PREFIX = 'XB'
    MODEL_ID_PREFIX = 'X'
    OPTUNA_ID_PREFIX = 'XO'
    TEST_ID_PREFIX = 'XT'

    # FEATURE SETS
    # Aligned with universal feature_engineering.py get_feature_sets().
    # 'full': None means "use all features returned by get_feature_columns()"
    # — the actual list is resolved at runtime by data_utils.load_prepared_data.
    #
    # Universal sets:
    #   full          – all generated features (dynamic; respects buy_pressure flag)
    #   technical     – balanced subset, no time/vol_regime (safe for short windows)
    #   technical_bp  – technical + buy_pressure (Binance full exports only)
    #   minimal       – smallest viable universal subset
    #
    # XGBoost-dedicated sets:
    #   xgb_momentum   – returns + MACD + RSI + SMA + vol_regime (+ opt. buy_pressure)
    #   xgb_volatility – vol-regime focused
    #   xgb_returns    – pure return-based, maximally portable
    FEATURE_SETS = {
        # Universal
        'full':           None,   # resolved at runtime via get_feature_columns()
        'technical':      None,   # resolved at runtime via get_feature_sets()
        'technical_bp':   None,
        'minimal':        None,
        # XGBoost-dedicated
        'xgb_momentum':   None,
        'xgb_volatility': None,
        'xgb_returns':    None,
        # Legacy aliases kept for backward compatibility
        'momentum':       None,   # maps → xgb_momentum at load time
        'volatility_focused': None,  # maps → xgb_volatility at load time
    }

    # FLATTENING METHODS
    FLATTENING_METHODS = ['lag', 'agg', 'hybrid']
    HYBRID_LAG_POINTS = [1, 2, 3, 5, 10, 24]
    HYBRID_WINDOWS = [6, 12, 24, 48]

    @staticmethod
    def load_optuna_params(params_path_or_dir, params_name='best'):
        """Load parameters from Optuna results"""
        if os.path.isdir(params_path_or_dir):
            params_path = os.path.join(params_path_or_dir, f'{params_name}_params.json')
        else:
            params_path = params_path_or_dir

        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Optuna params not found: {params_path}")

        with open(params_path, 'r') as f:
            data = json.load(f)

        params = {k: v for k, v in data.items() if not k.startswith('_')}
        metadata = data.get('_metadata', {})
        return params, metadata

    @staticmethod
    def train_with_optuna(data_path, optuna_path_or_dir, params_name='best',
                          seed=42, verbose=True, epochs=None):
        """Train model with Optuna parameters"""
        from xgb_train import train

        print("=" * 80)
        print("TRAINING WITH OPTUNA PARAMETERS")
        print("=" * 80)

        model, feature_scaler, config = train(
            data_path=data_path,
            load_optuna_params=optuna_path_or_dir
            if not os.path.isdir(optuna_path_or_dir)
            else os.path.join(optuna_path_or_dir, f'{params_name}_params.json'),
            seed=seed,
            verbose=verbose,
        )

        print(f"\nModel ID:  {config['model_id']}")
        print(f"Output:    {config['output_dir']}")
        print(f"Val MAE:   {config['val_metrics']['mae']:.6f}")
        print(f"Val R²:    {config['val_metrics']['r2']:.4f}")
        print(f"Dir Acc:   {config['val_metrics']['dir_acc']:.2f}%")

        return model, feature_scaler, config


# Create directories
for _dir in [XGBConfig.PREPARED_DATA_DIR, XGBConfig.MODELS_DIR,
             XGBConfig.LOGS_DIR, XGBConfig.RAW_DATA_DIR]:
    os.makedirs(_dir, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost with Optuna parameters')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--optuna_dir', type=str, default=None)
    parser.add_argument('--optuna_params', type=str, default=None)
    parser.add_argument('--params_name', type=str, default='best',
                        choices=['best', 'second_best', 'third_best'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    if args.optuna_params:
        optuna_path = args.optuna_params
    elif args.optuna_dir:
        optuna_path = args.optuna_dir
    else:
        raise ValueError("Provide either --optuna_dir or --optuna_params")

    XGBConfig.train_with_optuna(
        data_path=args.data_path,
        optuna_path_or_dir=optuna_path,
        params_name=args.params_name,
        seed=args.seed,
        verbose=not args.quiet,
        epochs=args.epochs
    )
