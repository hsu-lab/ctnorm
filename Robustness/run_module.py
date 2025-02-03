import os
import numpy as np
import pydicom
import sys
sys.path.append('/workspace/Robustness/Sybil')
from calculate_risk_scores import process_cases
def main(config, global_logger, session_path):
    current_mod = os.path.basename(os.path.dirname(__file__))
    datasets = config[current_mod]['input_datasets']
    parameters = config[current_mod]['metadata']
    for dataset in datasets:
        out_dir = os.path.join(session_path, current_mod, dataset)
        os.makedirs(out_dir, exist_ok=True)
        dataset_opt = config.get('Datasets', {}).get(dataset)
        if not dataset_opt:
            global_logger.error(f"Dataset '{dataset}' is missing in configuration under 'Datasets'.")
            continue
        input_file = dataset_opt.get('input_path')
        if not input_file:
            global_logger.error(f"Missing 'input_path' for dataset '{dataset}'.")
            continue
        process_cases(dataset, input_file, out_dir, global_logger,  parameters)
        output_file = os.path.join(out_dir, 'risk_scores.csv')