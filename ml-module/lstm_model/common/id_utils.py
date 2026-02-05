"""
ID management utilities for LSTM pipeline
"""
import os
import json


def get_next_id(base_dir, prefix, suffix_pattern):
    """Get next available ID with given prefix"""
    if not os.path.exists(base_dir):
        return f"{prefix}001"

    existing = [d for d in os.listdir(base_dir)
                if d.startswith(prefix) and suffix_pattern in d]
    if not existing:
        return f"{prefix}001"

    numbers = []
    for d in existing:
        try:
            num = int(d[len(prefix):len(prefix)+3])
            numbers.append(num)
        except:
            continue

    next_num = max(numbers) + 1 if numbers else 1
    return f"{prefix}{next_num:03d}"


def get_next_model_id(base_dir='lstm_models'):
    """Get next available model ID"""
    return get_next_id(base_dir, 'L', '_')


def get_next_optuna_id(base_dir='lstm_data'):
    """Get next available Optuna tuning ID"""
    return get_next_id(base_dir, 'O', '_optuna_tuning_')


def get_next_test_id(base_dir='lstm_data'):
    """Get next available test ID"""
    return get_next_id(base_dir, 'T', '_test_results_')


def extract_data_id_from_path(data_path):
    """Extract data ID from path or metadata"""
    original_path = data_path

    if not os.path.sep in data_path and not os.path.exists(data_path):
        candidate = os.path.join('lstm_prepared_data', data_path)
        if os.path.exists(candidate):
            data_path = candidate

    if os.path.isdir(data_path):
        metadata_path = os.path.join(data_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('data_id', 'UNKNOWN')

        basename = os.path.basename(data_path)
        if '_' in basename:
            parts = basename.split('_')
            if parts[0] and len(parts[0]) == 4 and parts[0][0].isalpha():
                return parts[0]

    elif data_path.endswith('.csv'):
        basename = os.path.basename(data_path)
        if '_' in basename:
            parts = basename.split('_')
            if parts[0] and len(parts[0]) == 4 and parts[0][0].isalpha():
                return parts[0]

    return 'UNKNOWN'


def extract_data_id(metadata_or_path):
    """Extract data ID from metadata dict or path"""
    if isinstance(metadata_or_path, dict):
        return metadata_or_path.get('data_id', 'UNKNOWN')
    elif isinstance(metadata_or_path, str):
        return extract_data_id_from_path(metadata_or_path)
    return 'UNKNOWN'


def extract_model_id(model_dir):
    """Extract model ID from model directory path"""
    basename = os.path.basename(model_dir)
    if '_' in basename and basename.startswith('L'):
        return basename.split('_')[0]
    return 'UNKNOWN'