"""
XGBoost Hyperparameter Tuning using Optuna
"""
import os
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from xgb_config import XGBConfig
from common.data_utils import load_data
from common.id_utils import get_next_optuna_id, extract_data_id_from_path

warnings.filterwarnings('ignore')


class TuningConfig:
    TRAIN_PATH      = '../data/'
    FEATURE_SET     = XGBConfig.DEFAULT_FEATURE_SET     # 'full'
    CUSTOM_FEATURES = None
    USE_SUBSET      = False
    SUBSET_SIZE     = 200000
    N_TRIALS        = 100
    TIMEOUT         = None
    N_JOBS          = 1
    ENABLE_PRUNING  = True
    PRUNING_WARMUP  = 10
    SEED            = 42
    FLATTENING_METHOD       = XGBConfig.DEFAULT_FLATTENING      # 'agg'
    FIXED_FLATTENING_METHOD = None
    LOOKBACK        = XGBConfig.DEFAULT_LOOKBACK        # 48
    HORIZON         = XGBConfig.DEFAULT_HORIZON         # 12
    TUNE_LOOKBACK   = False
    TUNE_HORIZON    = False
    SCALE_TARGET       = True
    TARGET_SCALER_TYPE = 'robust'
    OPTIMIZE_TARGET    = XGBConfig.OPTUNA_OPTIMIZE_TARGET   # 'val' | 'train'


def suggest_hyperparameters(trial):
    """Suggest hyperparameters for trial"""
    xgb_objective, extra_params = XGBConfig.get_xgb_objective()

    params = {
        'max_depth':          trial.suggest_int('max_depth', 3, 10),
        'min_child_weight':   trial.suggest_int('min_child_weight', 1, 7),
        'gamma':              trial.suggest_float('gamma', 0.0, 2.0),
        'subsample':          trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel':  trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'learning_rate':      trial.suggest_float('learning_rate', 0.00005, 0.5, log=True),
        'n_estimators':       trial.suggest_int('n_estimators', 400, 4000),
        'reg_alpha':          trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda':         trial.suggest_float('reg_lambda', 0.01, 5.0),
        'random_state':       TuningConfig.SEED,
        'n_jobs':             -1,
        'tree_method':        'hist',
        'objective':          xgb_objective,
        **extra_params,
    }

    params['lookback'] = (
        trial.suggest_categorical('lookback', [24, 36, 48, 72, 96, 120])
        if TuningConfig.TUNE_LOOKBACK else TuningConfig.LOOKBACK
    )
    params['horizon'] = (
        trial.suggest_categorical('horizon', [1, 3, 6, 12, 24])
        if TuningConfig.TUNE_HORIZON else TuningConfig.HORIZON
    )

    if TuningConfig.FIXED_FLATTENING_METHOD is not None:
        params['flattening_method'] = TuningConfig.FIXED_FLATTENING_METHOD
    else:
        params['flattening_method'] = trial.suggest_categorical(
            'flattening_method', ['lag', 'agg', 'hybrid']
        )

    return params


def objective(trial):
    """Objective function for Optuna"""
    try:
        params = suggest_hyperparameters(trial)

        data, _, _ = load_data(
            TuningConfig.TRAIN_PATH,
            flattening_method=params.get('flattening_method', TuningConfig.FLATTENING_METHOD),
            feature_set=TuningConfig.FEATURE_SET,
            custom_features=TuningConfig.CUSTOM_FEATURES,
            lookback=params.get('lookback', TuningConfig.LOOKBACK),
            horizon=params.get('horizon', TuningConfig.HORIZON),
            scale_target=TuningConfig.SCALE_TARGET,
            target_scaler_type=TuningConfig.TARGET_SCALER_TYPE,
        )

        X_train = data['X_train']
        y_train = data['y_train']
        X_val   = data['X_val']
        y_val   = data['y_val']
        target_scaler = data.get('target_scaler')

        if len(X_train) < 100 or len(X_val) < 50:
            raise optuna.TrialPruned("Insufficient data")

        # Strip non-XGBoost keys before passing to trainer
        train_params = {k: v for k, v in params.items()
                        if k not in ('lookback', 'horizon', 'flattening_method')}

        from common.training_utils import evaluate_for_tuning
        score = evaluate_for_tuning(
            X_train, y_train, X_val, y_val,
            train_params,
            early_stopping_rounds=XGBConfig.EARLY_STOPPING_ROUNDS,
            target_scaler=target_scaler,
            optimize_target=TuningConfig.OPTIMIZE_TARGET,
        )

        return score

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.TrialPruned()


def run_tuning(n_trials=None, timeout=None, study_name=None):
    """Run Optuna hyperparameter tuning"""
    if n_trials is None:
        n_trials = TuningConfig.N_TRIALS
    if timeout is None:
        timeout = TuningConfig.TIMEOUT

    data_id = extract_data_id_from_path(TuningConfig.TRAIN_PATH)
    os.makedirs(XGBConfig.LOGS_DIR, exist_ok=True)

    optuna_id  = get_next_optuna_id(XGBConfig.LOGS_DIR)
    date_short = datetime.now().strftime('%d%m')

    if study_name is None:
        study_name = f"{optuna_id}_xgb_optuna_{data_id}_{date_short}"

    output_dir = os.path.join(XGBConfig.LOGS_DIR, study_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'tuning.log'), encoding='utf-8'),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    xgb_objective, _ = XGBConfig.get_xgb_objective()

    logger.info("=" * 80)
    logger.info("XGBOOST HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    logger.info(f"Optuna ID:  {optuna_id}")
    logger.info(f"Data ID:    {data_id}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Trials:     {n_trials}")
    logger.info(f"Objective:  {xgb_objective}" + (f" (huber_slope={XGBConfig.HUBER_SLOPE})" if XGBConfig.OBJECTIVE == 'reg:pseudohubererror_large' else ""))
    logger.info(f"Optuna minimises: {TuningConfig.OPTIMIZE_TARGET.upper()} loss ({xgb_objective})")
    logger.info(f"Target scaling: {TuningConfig.SCALE_TARGET} ({TuningConfig.TARGET_SCALER_TYPE})")
    logger.info(f"Feature set: {TuningConfig.FEATURE_SET}")
    logger.info(f"Lookback:   {'TUNING' if TuningConfig.TUNE_LOOKBACK else f'FIXED ({TuningConfig.LOOKBACK})'}")
    logger.info(f"Horizon:    {'TUNING' if TuningConfig.TUNE_HORIZON else f'FIXED ({TuningConfig.HORIZON})'}")
    logger.info(f"Flattening: {'TUNING' if not TuningConfig.FIXED_FLATTENING_METHOD else f'FIXED ({TuningConfig.FIXED_FLATTENING_METHOD})'}")

    sampler = TPESampler(seed=TuningConfig.SEED)
    pruner  = (MedianPruner(n_startup_trials=5, n_warmup_steps=TuningConfig.PRUNING_WARMUP)
               if TuningConfig.ENABLE_PRUNING else None)

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=TuningConfig.N_JOBS,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == TrialState.PRUNED]

    logger.info(f"\nTotal: {len(study.trials)}, Complete: {len(completed)}, Pruned: {len(pruned)}")

    if not completed:
        logger.error("No trials completed!")
        return None, None

    best = study.best_trial
    score_label = f"{TuningConfig.OPTIMIZE_TARGET.upper()} {xgb_objective}"
    logger.info(f"\nBest trial {best.number}: {score_label}={best.value:.6f}")
    for k, v in sorted(best.params.items()):
        logger.info(f"  {k:20s}: {v}")

    # Save top-3 results
    sorted_trials = sorted(completed, key=lambda t: t.value)
    top_trials    = sorted_trials[:min(3, len(sorted_trials))]
    names         = ['best', 'second_best', 'third_best']

    for trial, name in zip(top_trials, names):
        params_to_save = trial.params.copy()
        if 'horizon' not in params_to_save:
            params_to_save['horizon'] = TuningConfig.HORIZON

        params_to_save['_metadata'] = {
            'optuna_id':          optuna_id,
            'data_id':            data_id,
            'best_value':         float(trial.value),
            'best_value_meaning': f"{TuningConfig.OPTIMIZE_TARGET}_loss ({xgb_objective})",
            'trial_number':       trial.number,
            'timestamp':          datetime.now().strftime('%Y%m%d_%H%M%S'),
            'feature_set':        TuningConfig.FEATURE_SET,
            'custom_features':    TuningConfig.CUSTOM_FEATURES,
            'scale_target':       TuningConfig.SCALE_TARGET,
            'target_scaler_type': TuningConfig.TARGET_SCALER_TYPE,
            'optimize_target':    TuningConfig.OPTIMIZE_TARGET,
            'xgb_objective':      xgb_objective,
        }

        with open(os.path.join(output_dir, f'{name}_params.json'), 'w') as f:
            json.dump(params_to_save, f, indent=2)
        logger.info(f"  [OK] {name}_params.json ({score_label}: {trial.value:.6f})")

    import joblib
    joblib.dump(study, os.path.join(output_dir, 'study.pkl'))
    study.trials_dataframe().to_csv(os.path.join(output_dir, 'trials.csv'), index=False)
    logger.info("  [OK] study.pkl, trials.csv")

    huber_note = f" (huber_slope={XGBConfig.HUBER_SLOPE})" if XGBConfig.OBJECTIVE == 'reg:pseudohubererror_large' else ""
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("XGBOOST OPTUNA TUNING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Optuna ID:  {optuna_id}\n")
        f.write(f"Data ID:    {data_id}\n")
        f.write(f"Objective:  {xgb_objective}{huber_note}\n")
        f.write(f"Minimises:  {TuningConfig.OPTIMIZE_TARGET.upper()} loss ({xgb_objective})\n")
        f.write(f"Feature set: {TuningConfig.FEATURE_SET}\n")
        f.write(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total:      {len(study.trials)}\n")
        f.write(f"Complete:   {len(completed)}\n\n")
        f.write("TOP 3 TRIALS:\n\n")
        for idx, (trial, name) in enumerate(zip(top_trials, names), 1):
            f.write(f"{idx}. {name.upper()}: trial={trial.number}, {score_label}={trial.value:.6f}\n")
            for k, v in sorted(trial.params.items()):
                f.write(f"   {k:20s}: {v}\n")
            f.write("\n")

    logger.info(f"\nNext step:")
    logger.info(f"  python xgb_train.py --data_path {TuningConfig.TRAIN_PATH} "
                f"--load_optuna_params {output_dir}/best_params.json")

    return study, best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--feature_set', type=str, default=None)
    parser.add_argument('--custom_features', type=str, nargs='+', default=None)
    parser.add_argument('--lookback', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--flattening_method', type=str, default=None,
                        choices=['lag', 'agg', 'hybrid'])
    parser.add_argument('--scale_target', action='store_true', default=None)
    parser.add_argument('--no_scale_target', dest='scale_target', action='store_false')
    parser.add_argument('--target_scaler_type', type=str, default=None,
                        choices=['standard', 'robust'])
    parser.add_argument('--tune_lookback', action='store_true')
    parser.add_argument('--tune_horizon', action='store_true')
    parser.add_argument('--optimize_target', type=str, default=None,
                        choices=['val', 'train'],
                        help="Which dataset loss Optuna minimises: 'val' (default) or 'train'")

    args = parser.parse_args()

    TuningConfig.TRAIN_PATH = args.data_path
    if args.feature_set:        TuningConfig.FEATURE_SET = args.feature_set
    if args.custom_features:    TuningConfig.CUSTOM_FEATURES = args.custom_features
    if args.lookback is not None:   TuningConfig.LOOKBACK = args.lookback
    if args.horizon is not None:    TuningConfig.HORIZON  = args.horizon
    if args.scale_target is not None:   TuningConfig.SCALE_TARGET = args.scale_target
    if args.target_scaler_type:         TuningConfig.TARGET_SCALER_TYPE = args.target_scaler_type
    TuningConfig.TUNE_LOOKBACK = args.tune_lookback
    TuningConfig.TUNE_HORIZON  = args.tune_horizon
    if args.optimize_target:    TuningConfig.OPTIMIZE_TARGET = args.optimize_target

    if args.flattening_method:
        TuningConfig.FIXED_FLATTENING_METHOD = args.flattening_method
        TuningConfig.FLATTENING_METHOD       = args.flattening_method

    try:
        run_tuning(n_trials=args.n_trials, timeout=args.timeout)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
