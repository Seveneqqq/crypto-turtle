"""
Common utilities for XGBoost model
"""

from .id_utils import (
    get_next_model_id,
    get_next_optuna_id,
    get_next_test_id,
    get_next_data_id,
    extract_model_id,
    extract_data_id,
    extract_data_id_from_path,
    detect_symbol,
)

from .metrics import (
    calculate_metrics,
    print_metrics_summary,
    interpret_metrics,
)

from .visualization import (
    plot_test_results,
    plot_feature_importance,
)

from .data_utils import (
    load_data,
    flatten_features,
    create_log_return_target,
)

__all__ = [
    'get_next_model_id',
    'get_next_optuna_id',
    'get_next_test_id',
    'get_next_data_id',
    'extract_model_id',
    'extract_data_id',
    'extract_data_id_from_path',
    'detect_symbol',
    'calculate_metrics',
    'print_metrics_summary',
    'interpret_metrics',
    'plot_test_results',
    'plot_feature_importance',
    'load_data',
    'flatten_features',
    'create_log_return_target',
]
