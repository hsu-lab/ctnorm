import os
import pickle
import numpy as np
import nibabel as nib
from ctnorm.Harmonization.data import read_csv
from alive_progress import alive_bar
import sys
import torch
import math
from ctnorm.Robustness.sybil import Serie, Sybil


def main(config, global_logger, session_path):
    current_mod = os.path.basename(os.path.dirname(__file__))
    datasets = config[current_mod]['input_datasets']
    models_param = config[current_mod]['param']
    model_type = models_param.get('model_type', 'sybil_ensemble')
    model = Sybil(model_type)

    for dataset in datasets:
        out_d = os.path.join(session_path, current_mod, dataset['name'])
        os.makedirs(out_d, exist_ok=True)
        global_logger.info('Sybil risk scores will be saved at: {}'.format(out_d))

        dataset_opt = config['Datasets'][dataset['name']]
        if dataset.get('in_uids', None):
            dataset_opt['in_uids'] = dataset['in_uids']
        if models_param.get('evaluate', None):
            need_evaluation = True
            df = read_csv(dataset_opt['in_uids'], ['uids', 'label', 'time_to_event'])
        else:
            need_evaluation = False
            df = read_csv(dataset_opt['in_uids'])

        if dataset.get('variability', None):
            load_from = dataset.get('load_from', None)
            if load_from:
                characterization_csv = os.path.join(config['Global']['session_base_path'], load_from, 'Characterization', dataset['name'], 'metadata_characterization.csv')
            else:
                characterization_csv = os.path.join(session_path, 'Characterization', dataset['name'], 'metadata_characterization.csv')

            base_df = read_csv(characterization_csv)
            var_category = {}
            for each_var in dataset['variability']:
                if each_var not in base_df.columns:
                    raise ValueError(f"Variability: {dataset['variability']['name']} not found in {characterization_csv}")
                _df = df.merge(base_df, on="uids", how="left")
                cur_gp = dict(tuple(_df.groupby(each_var)))
                var_gp = {}
                for key, gp in cur_gp.items():
                    var_gp[key] = gp
                var_category[each_var]= var_gp
        else:
            var_category = {'ALL':{'all_var':df}}

        for category in var_category:
            out_folder = os.path.join(out_d, category)
            os.makedirs(out_folder, exist_ok=True)
            each_category_var = var_category[category]

            for key, group in each_category_var.items():
                global_logger.info('Processing variability: {}'.format(key))
                all_series, all_scores = [], []
                gp_out_name = os.path.join(out_folder, '{}.pkl'.format(key))
                cur_gp_uids = group['uids'].to_list()
                if need_evaluation:
                    cur_gp_labels = group['label'].to_list()
                    # Check if there is at least binary label present; else AUC cannot be calculated
                    if not (1 in cur_gp_labels and 0 in cur_gp_labels):
                        global_logger.info('Skipping group {}: Binary label (1 and 0) not found for this group!'.format(key))
                        continue
                    cur_gp_tte = group['time_to_event'].to_list()
                
                for idx, gp_pth in enumerate(cur_gp_uids):
                    dicom_f = os.listdir(gp_pth)
                    dicom_f.sort()
                    dicom_fullpath = [os.path.join(gp_pth, f) for f in dicom_f]
                    if need_evaluation:
                        series = Serie(dicom_fullpath, label=int(cur_gp_labels[idx]), censor_time=int(cur_gp_tte[idx]))
                    else:
                        series = Serie(dicom_fullpath)
                    all_series.append(series)

                if need_evaluation:
                    results = model.evaluate(all_series)
                    with open(gp_out_name, "wb") as f:
                        pickle.dump(results, f)
                else:
                    scores = model.predict(all_series)
                    with open(gp_out_name, "wb") as f:
                        pickle.dump(all_scores, f)

