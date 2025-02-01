

import os
import pickle
import numpy as np
import pydicom
import logging
from .characterize import data_char

def main(config, global_logger, session_path):
    current_mod = os.path.basename(os.path.dirname(__file__))
    
    if current_mod not in config:
        global_logger.error(f"Configuration for module '{current_mod}' is missing.")
        return

    datasets = config[current_mod].get('input_datasets', [])
    
    if not datasets:
        global_logger.error("No datasets found in configuration.")
        return

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

        # Check if input file exists
        if not os.path.exists(input_file):
            global_logger.error(f"Input file '{input_file}' not found for dataset '{dataset}'.")
            continue

        # Check if input file is a CSV
        if not input_file.endswith('.csv'):
            global_logger.error(f"Invalid file format for dataset '{dataset}'. Expected CSV but got '{input_file}'.")
            continue

        bins = config[current_mod].get('bins', 64)
        voxel = config[current_mod].get('voxel', 1)
        metadata = config[current_mod].get('metadata', 1)
        if voxel == 0 and metadata == 0:
            global_logger.error(f"Characterization will not be performed for dataset '{dataset}' as both voxel and metadata are set to 0.") 
            continue
        # Run data characterization
        # data_char(input_file, out_dir, dataset, global_logger, bins, voxel, metadata)
        data_char(input_file, out_dir, dataset, global_logger, bins, voxel, metadata)
        # Check if both voxel and metadata are zero
        

        global_logger.info(f"Processing completed for dataset '{dataset}'.")








    

