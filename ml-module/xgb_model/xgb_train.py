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
from common.sample_weighting import (
    compute_sample_weights, make_custom_objective,
    make_dead_zone_objective, resolve_signal_threshold,
)
from common.log_utils import (
    log_run_header, log_train_session_header, log_data_summary,
    log_config_flags, log_signal_threshold, log_hyperparameters,
    log_train_val_metrics,
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
        # --- podejscie A: sample weights ---
        sample_weight_mode=None,
        sample_weight_alpha=None,
        sample_weight_min_w=None,
        sample_weight_percentile=None,
        sample_weight_low_w=None,
        # --- podejscie A: misdir penalty ---
        reweight_misdir=None,
        misdir_approach_a_variant=None,
        misdir_penalty=None,
        misdir_min_quantile=None,
        # --- podejscie B: custom objective ---
        use_custom_objective=None,
        custom_obj_sample_weights=None,
        custom_obj_misdir_penalty=None,
        custom_obj_huber_delta=None,
        # --- signal threshold ---
        signal_threshold_enabled=None,
        signal_threshold_mode=None,
        signal_threshold_value=None,
        signal_threshold_percentile=None,
        use_threshold_penalties=None,
        threshold_smooth_transition=None,
        threshold_smooth_width=None,
        custom_obj_type=None,
        dead_zone_weight=None,
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

    # Defaults dla podejscia A
    if sample_weight_mode is None:        sample_weight_mode        = XGBConfig.SAMPLE_WEIGHT_MODE
    if sample_weight_alpha is None:       sample_weight_alpha       = XGBConfig.SAMPLE_WEIGHT_ALPHA
    if sample_weight_min_w is None:       sample_weight_min_w       = XGBConfig.SAMPLE_WEIGHT_MIN_W
    if sample_weight_percentile is None:  sample_weight_percentile  = XGBConfig.SAMPLE_WEIGHT_PERCENTILE
    if sample_weight_low_w is None:       sample_weight_low_w       = XGBConfig.SAMPLE_WEIGHT_LOW_W
    if reweight_misdir is None:           reweight_misdir           = XGBConfig.REWEIGHT_MISDIR
    if misdir_approach_a_variant is None: misdir_approach_a_variant = XGBConfig.MISDIR_APPROACH_A_VARIANT
    if misdir_penalty is None:            misdir_penalty            = XGBConfig.MISDIR_PENALTY
    if misdir_min_quantile is None:       misdir_min_quantile       = XGBConfig.MISDIR_MIN_QUANTILE

    # Defaults dla podejscia B
    if use_custom_objective is None:       use_custom_objective       = XGBConfig.USE_CUSTOM_OBJECTIVE
    if custom_obj_sample_weights is None:  custom_obj_sample_weights  = XGBConfig.CUSTOM_OBJ_SAMPLE_WEIGHTS
    if custom_obj_misdir_penalty is None:  custom_obj_misdir_penalty  = XGBConfig.CUSTOM_OBJ_MISDIR_PENALTY
    if custom_obj_huber_delta is None:     custom_obj_huber_delta     = XGBConfig.CUSTOM_OBJ_HUBER_DELTA

    # Defaults dla signal threshold
    if signal_threshold_enabled is None:     signal_threshold_enabled     = XGBConfig.SIGNAL_THRESHOLD_ENABLED
    if signal_threshold_mode is None:        signal_threshold_mode        = XGBConfig.SIGNAL_THRESHOLD_MODE
    if signal_threshold_value is None:       signal_threshold_value       = XGBConfig.SIGNAL_THRESHOLD_VALUE
    if signal_threshold_percentile is None:  signal_threshold_percentile  = XGBConfig.SIGNAL_THRESHOLD_PERCENTILE
    if use_threshold_penalties is None:       use_threshold_penalties      = XGBConfig.USE_THRESHOLD_PENALTIES
    if threshold_smooth_transition is None:  threshold_smooth_transition  = XGBConfig.THRESHOLD_SMOOTH_TRANSITION
    if threshold_smooth_width is None:       threshold_smooth_width       = XGBConfig.THRESHOLD_SMOOTH_WIDTH
    if custom_obj_type is None:              custom_obj_type              = XGBConfig.CUSTOM_OBJ_TYPE
    if dead_zone_weight is None:             dead_zone_weight             = XGBConfig.DEAD_ZONE_WEIGHT

    # Walidacja -- blokada dublowania podejsc A i B
    XGBConfig.validate_approach_config()

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
        # 'reg:pseudohubererror_large' is a virtual name — map to real XGBoost objective
        if objective == 'reg:pseudohubererror_large':
            params['objective'] = 'reg:pseudohubererror'
            params['huber_slope'] = XGBConfig.HUBER_SLOPE
        else:
            params['objective'] = objective
    for k, v in overrides.items():
        if v is not None:
            params[k] = v

    log_run_header("XGBOOST TRAINING")

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

    # Raw y_train (dla threshold resolution i sample weights)
    if target_scaler is not None:
        y_train_raw = target_scaler.inverse_transform(y_train)
    else:
        y_train_raw = y_train

    # Resolve signal threshold (None gdy wyłączony)
    resolved_threshold = None
    if signal_threshold_enabled:
        resolved_threshold = resolve_signal_threshold(
            y_train_raw,
            mode=signal_threshold_mode,
            fixed_value=signal_threshold_value,
            percentile=signal_threshold_percentile,
        )

    # When USE_CUSTOM_OBJECTIVE=True the custom objective already applies the threshold
    # overlay internally — skip it here to avoid double-penalising below-threshold samples
    # (DMatrix weight × custom obj multiplier would compound to e.g. 0.1 × 0.1 = 0.01).
    _dmatrix_use_threshold = (
        use_threshold_penalties
        and resolved_threshold is not None
        and not use_custom_objective
    )
    sample_weight = compute_sample_weights(
        y_train_raw,
        mode=sample_weight_mode,
        alpha=sample_weight_alpha,
        min_w=sample_weight_min_w,
        percentile=sample_weight_percentile,
        low_w=sample_weight_low_w,
        use_threshold=_dmatrix_use_threshold,
        threshold=resolved_threshold or 0.001,
        smooth_transition=threshold_smooth_transition,
        smooth_width=threshold_smooth_width,
        threshold_low_w=dead_zone_weight,
    )

    # Stworz custom objective (podejscie B) jesli aktywna
    custom_objective = None
    if use_custom_objective:
        if custom_obj_type == 'dead_zone' and resolved_threshold is None:
            raise ValueError(
                "CUSTOM_OBJ_TYPE='dead_zone' wymaga aktywnego signal threshold "
                "(SIGNAL_THRESHOLD_ENABLED=True). Dead-zone objective nie moze dzialac "
                "bez progu — ustaw SIGNAL_THRESHOLD_ENABLED=True lub zmien CUSTOM_OBJ_TYPE."
            )
        if custom_obj_type == 'dead_zone':
            custom_objective = make_dead_zone_objective(
                target_scaler=target_scaler,
                signal_threshold=resolved_threshold,
                dead_zone_weight=dead_zone_weight,
                smooth_transition=threshold_smooth_transition,
                smooth_width=threshold_smooth_width,
                use_misdir_penalty=custom_obj_misdir_penalty,
                misdir_penalty=misdir_penalty,
                huber_delta=custom_obj_huber_delta,
            )
        else:  # 'weighted_huber'
            custom_objective = make_custom_objective(
                target_scaler=target_scaler,
                use_sample_weights=custom_obj_sample_weights,
                weight_mode=sample_weight_mode or 'percentile',
                weight_alpha=sample_weight_alpha,
                weight_min_w=sample_weight_min_w,
                weight_percentile=sample_weight_percentile,
                weight_low_w=sample_weight_low_w,
                use_misdir_penalty=custom_obj_misdir_penalty,
                misdir_penalty=misdir_penalty,
                misdir_min_quantile=misdir_min_quantile,
                huber_delta=custom_obj_huber_delta,
                use_threshold=use_threshold_penalties and resolved_threshold is not None,
                threshold=resolved_threshold,
                smooth_transition=threshold_smooth_transition,
                smooth_width=threshold_smooth_width,
                threshold_low_w=dead_zone_weight,
            )

    model_id   = get_next_model_id()
    date_short = datetime.now().strftime('%d%m')
    run_name   = f"{model_id}_{symbol}_xgb_model_{data_id}_{date_short}"
    output_dir = os.path.join('xgb_models', run_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    log_train_session_header(logger, symbol, horizon, model_id, data_id, output_dir)
    log_data_summary(
        features, flat_features, flattening_method, lookback, horizon,
        len(X_train), len(X_val), logger=logger,
    )
    log_config_flags(
        target_scaler, winsorize_target, winsorize_percentile,
        sample_weight_mode, reweight_misdir, misdir_approach_a_variant,
        use_custom_objective, custom_obj_type, custom_obj_sample_weights,
        custom_obj_misdir_penalty, custom_obj_huber_delta, logger=logger,
    )
    log_signal_threshold(
        signal_threshold_enabled, resolved_threshold, signal_threshold_mode,
        use_threshold_penalties, threshold_smooth_transition, threshold_smooth_width,
        clip_bounds=clip_bounds, logger=logger,
    )
    use_best_iteration = XGBConfig.USE_BEST_ITERATION
    log_hyperparameters(logger, params, early_stopping_rounds, use_best_iteration)

    model, best_iteration, best_score, train_metrics, val_metrics = _train_model(
        X_train, y_train, X_val, y_val,
        params, early_stopping_rounds,
        logger, verbose,
        target_scaler=target_scaler,
        use_best_iteration=use_best_iteration,
        sample_weight=sample_weight,
        reweight_misdir=reweight_misdir,
        misdir_approach_a_variant=misdir_approach_a_variant,
        misdir_penalty=misdir_penalty,
        misdir_min_quantile=misdir_min_quantile,
        custom_objective=custom_objective,
        signal_threshold=resolved_threshold,
    )

    log_train_val_metrics(logger, train_metrics, val_metrics)

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
        'scaler_feature_cols': list(feature_scaler.feature_names_in_)
                               if hasattr(feature_scaler, 'feature_names_in_')
                               else None,
        'sample_weight_mode':        sample_weight_mode,
        'sample_weight_alpha':       sample_weight_alpha       if sample_weight_mode else None,
        'sample_weight_min_w':       sample_weight_min_w       if sample_weight_mode == 'abs_clip' else None,
        'sample_weight_percentile':  sample_weight_percentile  if sample_weight_mode == 'percentile' else None,
        'sample_weight_low_w':       sample_weight_low_w       if sample_weight_mode == 'percentile' else None,
        'reweight_misdir':           reweight_misdir,
        'misdir_approach_a_variant': misdir_approach_a_variant if reweight_misdir else None,
        'misdir_penalty':            misdir_penalty             if reweight_misdir else None,
        'misdir_min_quantile':       misdir_min_quantile        if reweight_misdir else None,
        'use_custom_objective':      use_custom_objective,
        'custom_obj_type':           custom_obj_type             if use_custom_objective else None,
        'custom_obj_sample_weights': custom_obj_sample_weights  if use_custom_objective else None,
        'custom_obj_misdir_penalty': custom_obj_misdir_penalty  if use_custom_objective else None,
        'custom_obj_huber_delta':    custom_obj_huber_delta     if use_custom_objective else None,
        'signal_threshold_enabled':     signal_threshold_enabled,
        'signal_threshold_mode':        signal_threshold_mode       if signal_threshold_enabled else None,
        'signal_threshold_value':       signal_threshold_value      if signal_threshold_enabled else None,
        'signal_threshold_resolved':    resolved_threshold,
        'use_threshold_penalties':      use_threshold_penalties,
        'threshold_smooth_transition':  threshold_smooth_transition if use_threshold_penalties else None,
        'threshold_smooth_width':       threshold_smooth_width      if use_threshold_penalties else None,
        'dead_zone_weight':             dead_zone_weight            if custom_obj_type == 'dead_zone' else None,
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
                        choices=['lag', 'agg', 'hybrid', 'signal', 'none'])
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
    # podejscie A: sample weights
    parser.add_argument('--sample_weight_mode', type=str, default=None,
                        choices=['abs', 'abs_clip', 'rank', 'percentile'],
                        help='Sample weighting scheme (None = uniform)')
    parser.add_argument('--sample_weight_alpha', type=float, default=None,
                        help='Exponent for abs/abs_clip modes (default 1.0)')
    parser.add_argument('--sample_weight_min_w', type=float, default=None,
                        help='Weight floor fraction for abs_clip (default 0.1)')
    parser.add_argument('--sample_weight_percentile', type=float, default=None,
                        help='Threshold percentile for percentile mode (default 70.0)')
    parser.add_argument('--sample_weight_low_w', type=float, default=None,
                        help='Weight for below-threshold samples in percentile mode (default 0.1)')
    # podejscie A: misdir penalty
    parser.add_argument('--reweight_misdir', action='store_true', default=None,
                        help='Enable mis-direction penalty (second training run)')
    parser.add_argument('--misdir_approach_a_variant', type=str, default=None,
                        choices=['retrain', 'boost'],
                        help='retrain=full restart, boost=continue from checkpoint')
    parser.add_argument('--misdir_penalty', type=float, default=None,
                        help='Multiplier for mis-directed large-move samples (default 2.0)')
    parser.add_argument('--misdir_min_quantile', type=float, default=None,
                        help='|y| quantile threshold to qualify for penalty (default 0.7)')
    # podejscie B: custom objective
    parser.add_argument('--use_custom_objective', action='store_true', default=None,
                        help='Enable custom objective (replaces params objective string)')
    parser.add_argument('--custom_obj_sample_weights', action='store_true', default=None,
                        help='Enable sample weight component in custom objective')
    parser.add_argument('--no_custom_obj_sample_weights',
                        dest='custom_obj_sample_weights', action='store_false')
    parser.add_argument('--custom_obj_misdir_penalty', action='store_true', default=None,
                        help='Enable misdir penalty component in custom objective')
    parser.add_argument('--no_custom_obj_misdir_penalty',
                        dest='custom_obj_misdir_penalty', action='store_false')
    parser.add_argument('--custom_obj_huber_delta', type=str, default=None,
                        help="Pseudo-Huber delta: 'auto' (default) or float value")
    # signal threshold
    parser.add_argument('--signal_threshold_enabled', action='store_true', default=None,
                        help='Enable signal threshold metrics and optional penalties')
    parser.add_argument('--signal_threshold_mode', type=str, default=None,
                        choices=['fixed', 'empirical'],
                        help='Signal threshold mode (default: fixed)')
    parser.add_argument('--signal_threshold_value', type=float, default=None,
                        help='Fixed signal threshold value (default 0.001)')
    parser.add_argument('--signal_threshold_percentile', type=float, default=None,
                        help='Percentile of |y| for empirical threshold (default 50.0)')
    parser.add_argument('--use_threshold_penalties', action='store_true', default=None,
                        help='Enable threshold-based penalties in objective/weights')
    parser.add_argument('--threshold_smooth_transition', action='store_true', default=None,
                        help='Enable smooth sigmoid transition around threshold')
    parser.add_argument('--threshold_smooth_width', type=float, default=None,
                        help='Sigmoid width for smooth transition (default 2.0)')
    parser.add_argument('--custom_obj_type', type=str, default=None,
                        choices=['weighted_huber', 'dead_zone'],
                        help='Custom objective type (default: weighted_huber)')
    parser.add_argument('--dead_zone_weight', type=float, default=None,
                        help='Gradient weight for |y| < threshold in dead_zone mode (default 0.1)')

    args = parser.parse_args()

    # Parse huber_delta: 'auto' stays as string, otherwise convert to float
    parsed_huber_delta = args.custom_obj_huber_delta
    if parsed_huber_delta is not None and parsed_huber_delta != 'auto':
        try:
            parsed_huber_delta = float(parsed_huber_delta)
        except ValueError:
            raise ValueError(
                f"--custom_obj_huber_delta must be 'auto' or a float, got: {parsed_huber_delta!r}"
            )

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
        sample_weight_mode=args.sample_weight_mode,
        sample_weight_alpha=args.sample_weight_alpha,
        sample_weight_min_w=args.sample_weight_min_w,
        sample_weight_percentile=args.sample_weight_percentile,
        sample_weight_low_w=args.sample_weight_low_w,
        reweight_misdir=args.reweight_misdir,
        misdir_approach_a_variant=args.misdir_approach_a_variant,
        misdir_penalty=args.misdir_penalty,
        misdir_min_quantile=args.misdir_min_quantile,
        use_custom_objective=args.use_custom_objective,
        custom_obj_sample_weights=args.custom_obj_sample_weights,
        custom_obj_misdir_penalty=args.custom_obj_misdir_penalty,
        custom_obj_huber_delta=parsed_huber_delta,
        signal_threshold_enabled=args.signal_threshold_enabled,
        signal_threshold_mode=args.signal_threshold_mode,
        signal_threshold_value=args.signal_threshold_value,
        signal_threshold_percentile=args.signal_threshold_percentile,
        use_threshold_penalties=args.use_threshold_penalties,
        threshold_smooth_transition=args.threshold_smooth_transition,
        threshold_smooth_width=args.threshold_smooth_width,
        custom_obj_type=args.custom_obj_type,
        dead_zone_weight=args.dead_zone_weight,
    )
