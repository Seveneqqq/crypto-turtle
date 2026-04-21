"""
ID management utilities for Random Forest pipeline
-- port from xgb_model/common/id_utils.py with RF-specific prefixes --
"""
import os
import json


def get_next_id(base_dir, prefix, suffix_pattern):
    """Get next available ID with given prefix"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return f"{prefix}001"

    existing = [d for d in os.listdir(base_dir)
                if d.startswith(prefix) and suffix_pattern in d]
    if not existing:
        return f"{prefix}001"

    numbers = []
    for d in existing:
        try:
            num_str = d[len(prefix):len(prefix) + 3]
            numbers.append(int(num_str))
        except (ValueError, IndexError):
            continue

    next_num = max(numbers) + 1 if numbers else 1
    return f"{prefix}{next_num:03d}"


def get_next_model_id(base_dir='rf_models'):
    """Get next available RF model ID (prefix: R)"""
    return get_next_id(base_dir, 'R', '_')


def get_next_optuna_id(base_dir='rf_data'):
    """Get next available Optuna tuning ID (prefix: RO)"""
    return get_next_id(base_dir, 'RO', '_optuna_')


def get_next_test_id(base_dir='rf_data'):
    """Get next available test ID (prefix: RT)"""
    return get_next_id(base_dir, 'RT', '_test_results_')


def get_next_data_id(base_dir='rf_prepared_data'):
    """Get next available prepared data ID (prefix: RB)"""
    return get_next_id(base_dir, 'RB', '_prep_')


def extract_model_id(model_dir):
    """Extract model ID from model directory path"""
    basename = os.path.basename(model_dir)
    if '_' in basename and basename.startswith('R'):
        parts = basename.split('_')
        if parts[0] and len(parts[0]) >= 4:   # R001 or longer
            return parts[0]
    return 'UNKNOWN'


def extract_data_id_from_path(data_path):
    """Extract data ID from path or metadata"""
    if os.path.isdir(data_path):
        metadata_path = os.path.join(data_path, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f).get('data_id', 'UNKNOWN')
            except Exception:
                pass
        basename = os.path.basename(data_path)
        if '_' in basename:
            parts = basename.split('_')
            if parts[0] and len(parts[0]) >= 4 and parts[0].startswith('RB'):
                return parts[0]

    elif os.path.isfile(data_path):
        basename = os.path.basename(data_path)
        if '_' in basename:
            parts = basename.split('_')
            if parts[0] and len(parts[0]) >= 4 and parts[0].startswith('RB'):
                return parts[0]

    return 'UNKNOWN'


def extract_data_id(metadata_or_path):
    """Extract data ID from metadata dict or path"""
    if isinstance(metadata_or_path, dict):
        return metadata_or_path.get('data_id', 'UNKNOWN')
    elif isinstance(metadata_or_path, str):
        return extract_data_id_from_path(metadata_or_path)
    return 'UNKNOWN'


def detect_symbol(filename):
    """Detect cryptocurrency symbol from filename"""
    filename_lower = filename.lower()
    symbols = {
        'btc': 'btc', 'bitcoin': 'btc',
        'eth': 'eth', 'ethereum': 'eth',
        'bnb': 'bnb', 'binance': 'bnb',
        'sol': 'sol', 'solana': 'sol',
        'ada': 'ada', 'cardano': 'ada',
        'xrp': 'xrp', 'ripple': 'xrp',
        'doge': 'doge', 'dogecoin': 'doge',
        'dot': 'dot', 'polkadot': 'dot',
        'matic': 'matic', 'polygon': 'matic',
        'avax': 'avax', 'avalanche': 'avax',
    }
    for key, symbol in symbols.items():
        if key in filename_lower:
            return symbol
    return 'crypto'
