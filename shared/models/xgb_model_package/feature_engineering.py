"""
Feature Engineering bridge for xgb_model_package.

Locates and loads the root feature_engineering.py from one of three locations,
checked in order:
  1. Parent directory of this file (development / standalone)
     e.g. claude_xgb/feature_engineering.py
  2. crypto-turtle/shared/models/feature_engineering.py
  3. crypto-turtle/ml-module/feature_engineering.py

All public symbols from the root module are re-exported here so that
xgb_model_package.py can simply do:
    from feature_engineering import prepare_features, select_features
"""
import os
import importlib.util
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_project_root(marker: str = 'crypto-turtle'):
    """Walk up the directory tree looking for a folder named *marker*."""
    current = _HERE
    for _ in range(12):
        if os.path.basename(current) == marker:
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def _candidate_paths():
    candidates = []

    # 1. Parent of xgb_model_package/ — covers claude_xgb/feature_engineering.py
    parent_fe = os.path.join(os.path.dirname(_HERE), 'feature_engineering.py')
    candidates.append(parent_fe)

    # 2 & 3 — only reachable if we can find the crypto-turtle project root
    root = _find_project_root('crypto-turtle')
    if root:
        candidates.append(os.path.join(root, 'shared', 'models', 'feature_engineering.py'))
        candidates.append(os.path.join(root, 'ml-module', 'feature_engineering.py'))

    return candidates


def _load_feature_engineering():
    candidates = _candidate_paths()
    for path in candidates:
        if not os.path.exists(path):
            continue
        # Guard: never load this bridge file itself
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue
        spec = importlib.util.spec_from_file_location('_root_feature_engineering', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['_root_feature_engineering'] = module
        spec.loader.exec_module(module)
        return module

    raise FileNotFoundError(
        "feature_engineering.py not found in any of the following locations:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\n\nMake sure feature_engineering.py is accessible from xgb_model_package."
    )


_fe = _load_feature_engineering()

# Re-export everything the package needs
prepare_features    = _fe.prepare_features
get_feature_columns = _fe.get_feature_columns
get_feature_sets    = _fe.get_feature_sets
select_features     = _fe.select_features
get_binance_columns = _fe.get_binance_columns
