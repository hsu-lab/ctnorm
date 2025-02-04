

import os
import pickle
import numpy as np
import pydicom
import logging
from .characterize import data_char

def main(config, global_logger, session_path):
    current_mod = os.path.basename(os.path.dirname(__file__))
    
    if current_mod not in config:
        raise RuntimeError(f"Configuration for module '{current_mod}' is missing.")

    datasets = config[current_mod].get('input_datasets', [])
    if not datasets:
        raise RuntimeError("No datasets found in configuration.")

    for dataset in datasets:
        out_dir = os.path.join(session_path, current_mod, dataset['name'])
        os.makedirs(out_dir, exist_ok=True)

        dataset_opt = config.get('Datasets', {}).get(dataset['name'])
        if not dataset_opt:
            raise RuntimeError(f"Dataset '{dataset}' is missing in configuration under 'Datasets'.")

        input_file = dataset_opt.get('in_uids')
        if not input_file:
            global_logger.error(f"Missing 'input_path' for dataset '{dataset['name']}'.")
            continue

        # Check if input file exists
        if not os.path.exists(input_file):
            raise RuntimeError(f"Input file '{input_file}' not found for dataset '{dataset['name']}'.")

        # Check if input file is a CSV
        if not input_file.endswith('.csv'):
            raise RuntimeError(f"Invalid file format for dataset '{dataset['name']}'. Expected CSV but got '{input_file}'.")

        bins = config[current_mod].get('bins', 64)
        voxel = config[current_mod].get('voxel', 1)
        metadata = config[current_mod].get('metadata', 1)
        if voxel == 0 and metadata == 0:
            raise RuntimeError(f"Characterization will not be performed for dataset '{dataset['name']}' as both voxel and metadata are set to 0.")

        # Run data characterization
        # data_char(input_file, out_dir, dataset, global_logger, bins, voxel, metadata)
        data_char(input_file, out_dir, dataset, global_logger, bins, voxel, metadata)
        # Check if both voxel and metadata are zero
        global_logger.info(f"Processing completed for dataset '{dataset['name']}'.")








    

