import os
import pickle
from .data import create_dataset, create_dataloader
from .models import create_model
from .helpers import fileout
import numpy as np
import nibabel as nib
from alive_progress import alive_bar
import sys


def main(config, global_logger, session_path):
    current_mod = os.path.basename(os.path.dirname(__file__))
    datasets = config[current_mod]['input_datasets']
    models = config[current_mod]['models']
    models_param = config[current_mod]['param']
    mode = config[current_mod]['mode']

    if mode == 'test':
        for model in models:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cnf = os.path.join(base_dir, 'model_configs', model.lower()+'.pkl')
            # load model-specific params
            with open(cnf, 'rb') as file:
                model_opt = pickle.load(file)
            if mode == 'train':
                model_opt['is_train'] = True
            else:
                model_opt['is_train'] = False
            model_opt['network_G']['scale'] = models_param.get('scale', 1.0)
            model_opt['gpu_id'] = models_param['gpu_id']
            model_opt['path'] = {'pretrain_model_G':models[model]}
            model_opt['dataset_opt'] = {
                'tile_xy': models_param.get('tile_z', 512),
                'tile_z': models_param.get('tile_z', 32),
                'z_overlap': models_param.get('z_overlap', 4),
                'scale': models_param.get('scale', 1.0)
            }
            # create model and convert to fp16 for faster computation
            dl_model = create_model(model_opt)
            dl_model.half()

            for dataset in datasets:
                out_d = os.path.join(session_path, current_mod, dataset['name'], mode)
                os.makedirs(out_d, exist_ok=True)
                # load dataset-specific config
                dataset_opt = config['Datasets'][dataset['name']]
                if dataset.get('subset', None):
                    dataset_opt['uids'] = dataset.get('subset', None)
                dataset_opt['is_train'] = model_opt['is_train']
                dataset_opt['out'] = out_d
                test_set = create_dataset(dataset_opt)
                test_loader = create_dataloader(test_set, dataset_opt)
                total_iterations = len(test_loader)

                with alive_bar(total_iterations, title='Processing files') as bar:
                    for i, data in enumerate(test_loader):
                        need_HR = False if test_loader.dataset.opt.get('dataroot_HR') is None else True
                        dl_model.feed_test_data(data, need_HR=need_HR)
                        dl_model.test(data)
                        visuals = dl_model.get_current_visuals(data, need_HR=need_HR)
                        sr_vol = dl_model.tensor2img(visuals['SR'], out_type=np.int16)
                        # save output
                        fileout.save_volume(sr_vol, out_type=models_param['out_dtype'], out_dir=os.path.join(out_d, data['uid'][0]), m_type='Volume', f_name='{}--{}'.format(model, data['uid'][0]), target_scale=models_param.get('scale', None))
                        bar()

    elif mode == 'train':
        pass
    
    else:
        global_logger.info(f'Mode {mode} not implemented!')