"""
Training utilities for XGBoost models
"""
import os
import json
import logging

import numpy as np
import xgboost as xgb
import joblib

from common.metrics import calculate_metrics


def set_seed(seed=42):
    np.random.seed(seed)


def setup_logger(output_dir, name='xgb_training'):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(output_dir, 'training.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _inverse_if_scaled(y, target_scaler):
    """Inverse transform if scaler provided, else return as-is"""
    if target_scaler is not None:
        return target_scaler.inverse_transform(y)
    return y


def _compute_metrics(y_true_scaled, y_pred_scaled, target_scaler):
    """Compute metrics in original log-return space"""
    y_true = _inverse_if_scaled(y_true_scaled, target_scaler)
    y_pred = _inverse_if_scaled(y_pred_scaled, target_scaler)
    return calculate_metrics(y_true, y_pred)


def train_model(X_train, y_train, X_val, y_val,
                params, early_stopping_rounds,
                logger, verbose=True, target_scaler=None,
                use_best_iteration=False):
    """
    Train XGBoost with holdout validation.
    X_train/X_val are separate — no leakage into early stopping.
    y_train/y_val are already scaled if target_scaler is provided.

    use_best_iteration=False (default): predict and save from last iteration
    use_best_iteration=True:            predict and save from best_iteration checkpoint

    Early stopping always monitors validation loss (last entry in evals list).

    Returns: (model, best_iteration, best_score, train_metrics, val_metrics)
    """
    if target_scaler is not None:
        logger.info(f"Target scaling: ENABLED ({target_scaler.method})")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    # Early stopping always monitors the LAST entry — keep val last
    evals_list = [(dtrain, 'train'), (dval, 'validation')]

    model = xgb.train(
        params, dtrain,
        num_boost_round=params.get('n_estimators', 1000),
        evals=evals_list,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval_interval(verbose),
    )

    n_total  = model.num_boosted_rounds()
    best_iter = model.best_iteration  # 0-based index of best round

    if use_best_iteration:
        iter_range = (0, best_iter + 1)
        saved_at = f"best_iteration ({best_iter})"
    else:
        iter_range = (0, n_total)
        saved_at = f"last_iteration ({n_total - 1})"

    logger.info(f"Best iteration: {best_iter}, Total rounds: {n_total}, "
                f"Saving from: {saved_at}")

    # Validate X shape before predict
    if X_train.shape[1] != X_val.shape[1]:
        logger.warning(f"[SHAPE WARNING] X_train cols={X_train.shape[1]} != X_val cols={X_val.shape[1]}")

    y_train_pred = model.predict(dtrain, iteration_range=iter_range)
    y_val_pred   = model.predict(dval,   iteration_range=iter_range)

    train_metrics = _compute_metrics(y_train, y_train_pred, target_scaler)
    val_metrics   = _compute_metrics(y_val,   y_val_pred,   target_scaler)

    if use_best_iteration:
        model.set_attr(saved_iteration=str(best_iter + 1))
    else:
        model.set_attr(saved_iteration=str(n_total))

    return model, best_iter, model.best_score, train_metrics, val_metrics


def verbose_eval_interval(verbose):
    """Return verbose_eval value for xgb.train"""
    if verbose is True:
        return 60
    if isinstance(verbose, int):
        return verbose
    return False


def evaluate_for_tuning(X_train, y_train, X_val, y_val,
                        params, early_stopping_rounds,
                        target_scaler=None,
                        optimize_target='val'):
    """
    Evaluate hyperparameters for Optuna.
    Returns the XGBoost native loss (objective function score) on the chosen
    dataset — val or train — so Optuna minimises the same loss that XGBoost
    itself optimises during tree construction.

    Args:
        optimize_target: 'val' (default) or 'train'
            'val'   → return XGBoost eval score on validation set
            'train' → return XGBoost eval score on training set
    """
    if optimize_target not in ('val', 'train'):
        raise ValueError(f"optimize_target must be 'val' or 'train', got: {optimize_target!r}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    evals_result = {}
    bst = xgb.train(
        params, dtrain,
        num_boost_round=params.get('n_estimators', 1000),
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
        evals_result=evals_result,
    )

    # evals_result keys: 'train', 'validation'
    # each maps metric_name -> list of values per round
    if optimize_target == 'val':
        metric_history = evals_result.get('validation', {})
    else:
        metric_history = evals_result.get('train', {})

    if not metric_history:
        # Fallback: sklearn MAE on val
        y_pred = bst.predict(dval if optimize_target == 'val' else dtrain)
        y_ref  = y_val if optimize_target == 'val' else y_train
        metrics = _compute_metrics(y_ref, y_pred, target_scaler)
        return metrics['mae']

    # Take the last (final) value of the XGBoost native loss
    metric_name  = next(iter(metric_history))
    final_score  = metric_history[metric_name][-1]
    return float(final_score)


def save_feature_importance(model, flat_features, output_dir, logger):
    """
    Save feature importance (gain, weight, cover) to feature_importance.json.
    Keys from get_score() may be positional ('f0','f1',...) when DMatrix was
    created without feature names — remap them to actual flat_feature names.
    """
    importance_types = ['gain', 'weight', 'cover']
    result = {}

    for imp_type in importance_types:
        raw_scores = model.get_score(importance_type=imp_type)

        # Remap positional keys (f0, f1, ...) → actual feature names
        remapped = {}
        for k, v in raw_scores.items():
            if k.startswith('f') and k[1:].isdigit():
                idx = int(k[1:])
                name = flat_features[idx] if idx < len(flat_features) else k
            else:
                name = k
            remapped[name] = float(v)

        # Fill missing features with 0.0 and sort by score descending
        full_scores = {f: remapped.get(f, 0.0) for f in flat_features}
        result[imp_type] = dict(
            sorted(full_scores.items(), key=lambda x: x[1], reverse=True)
        )

    # Summary: top-20 by gain
    top20_gain = list(result['gain'].items())[:20]
    result['top20_gain'] = {k: v for k, v in top20_gain}

    importance_path = os.path.join(output_dir, 'feature_importance.json')
    with open(importance_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("  ✓ feature_importance.json")
    return result


def save_xgb_model(model, feature_scaler, config_dict, output_dir, logger, target_scaler=None):
    """Save XGBoost model, scalers, config, feature importance and summary"""
    logger.info("\nSAVING MODEL")

    model.save_model(os.path.join(output_dir, 'xgb_model.json'))
    logger.info("  ✓ xgb_model.json")

    joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
    logger.info("  ✓ feature_scaler.pkl")

    if target_scaler is not None:
        joblib.dump(target_scaler, os.path.join(output_dir, 'target_scaler.pkl'))
        config_dict['target_scaler_params'] = target_scaler.get_params()
        logger.info("  ✓ target_scaler.pkl")

    # Feature importance JSON
    flat_features = config_dict.get('flat_features', [])
    if flat_features:
        save_feature_importance(model, flat_features, output_dir, logger)

    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info("  ✓ model_config.json")

    val_metrics = config_dict['val_metrics']
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("XGBOOST MODEL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model ID:   {config_dict['model_id']}\n")
        f.write(f"Data ID:    {config_dict['data_id']}\n")
        f.write(f"Symbol:     {config_dict['symbol'].upper()}\n")
        f.write(f"Horizon:    t+{config_dict['horizon']}\n")
        f.write(f"Features:   {config_dict['n_features']} → {config_dict['n_flat_features']} flat\n")
        f.write(f"Flattening: {config_dict['flattening_method']}\n")
        f.write(f"Target scaling: {'ENABLED (' + target_scaler.method + ')' if target_scaler else 'DISABLED'}\n")
        f.write(f"Timestamp:  {config_dict['timestamp']}\n\n")
        f.write("HYPERPARAMETERS:\n")
        for key, value in sorted(config_dict['params'].items()):
            f.write(f"  {key:20s}: {value}\n")
        f.write(f"  early_stopping:      {config_dict.get('early_stopping_rounds', 'N/A')}\n\n")
        f.write("TRAIN METRICS (log return):\n")
        train_m = config_dict['train_metrics']
        f.write(f"  MAE:      {train_m['mae']:.6f}\n")
        f.write(f"  RMSE:     {train_m['rmse']:.6f}\n")
        f.write(f"  R²:       {train_m['r2']:.4f}\n")
        f.write(f"  Dir Acc:  {train_m['dir_acc']:.2f}%\n\n")
        f.write("VALIDATION METRICS (log return):\n")
        f.write(f"  MAE:      {val_metrics['mae']:.6f}\n")
        f.write(f"  RMSE:     {val_metrics['rmse']:.6f}\n")
        f.write(f"  R²:       {val_metrics['r2']:.4f}\n")
        f.write(f"  Dir Acc:  {val_metrics['dir_acc']:.2f}%\n")
    logger.info("  ✓ summary.txt")


def print_final_summary(model_id, output_dir, train_metrics, val_metrics, logger):
    """Print final training summary"""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model ID:      {model_id}")
    logger.info(f"Output:        {output_dir}")
    logger.info(f"Train MAE:     {train_metrics['mae']:.6f}")
    logger.info(f"Train R²:      {train_metrics['r2']:.4f}")
    logger.info(f"Val MAE:       {val_metrics['mae']:.6f}")
    logger.info(f"Val R²:        {val_metrics['r2']:.4f}")
    logger.info(f"Val Dir Acc:   {val_metrics['dir_acc']:.2f}%")
    logger.info("=" * 80)
