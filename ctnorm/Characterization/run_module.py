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
        
        in_dtype = dataset_opt.get('in_dtype')
        if not in_dtype in ['.dcm']:
            raise RuntimeError(f"Dataset '{dataset}' type must be dicom!")

        input_file = dataset_opt.get('in_uids')
        if not input_file:
            raise RuntimeError(f"Missing 'input_path' for dataset '{dataset['name']}'")

        # Check if input file exists
        if not os.path.exists(input_file):
            raise RuntimeError(f"Input file '{input_file}' not found for dataset '{dataset['name']}'")

        # Check if input file is a CSV
        if not input_file.endswith('.csv'):
            raise RuntimeError(f"Invalid file format for dataset '{dataset['name']}'. Expected CSV but got '{input_file}'")
        metrics = config[current_mod].get('metrics', {'voxel':'all', 'metadata':'all'})
        params = config[current_mod].get('params' )

        data_char(input_file, in_dtype, out_dir, dataset, global_logger, params, metrics)
        # Check if both voxel and metadata are zero
        global_logger.info(f"Processing completed for dataset '{dataset['name']}'.")

