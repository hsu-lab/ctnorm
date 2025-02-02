import os
import pickle
from .data import create_dataset, create_dataloader
from .models import create_model
from .helpers import helper_func
import numpy as np
import nibabel as nib
from alive_progress import alive_bar
import sys
import torch
import math


def main(config, global_logger, session_path):
    current_mod = os.path.basename(os.path.dirname(__file__))
    datasets = config[current_mod]['input_datasets']
    models = config[current_mod]['models']
    models_param = config[current_mod]['param']
    mode = config[current_mod]['mode']
    data_specific_opt = {
        'tile_xy': models_param.get('tile_xy', 64),
        'tile_z': models_param.get('tile_z', 32),
        'z_overlap': models_param.get('z_overlap', 4),
        'scale': models_param.get('scale', 1)
    }

    for model in models:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cnf = os.path.join(base_dir, 'model_configs', model['name'].lower()+'.pkl')
        # Load model-specific base params
        with open(cnf, 'rb') as file:
            model_opt = pickle.load(file)
        if mode == 'train':
            # # Override base model parameters if specified
            model_opt['is_train'] = True
            model_opt['path'] = {}
            model_opt['network_G']['in_nc'] = model.get('model_config', {}).get('nc_in', 1)
            model_opt['network_G']['nf'] = model.get('model_config', {}).get('nf', 64)
            model_opt['network_G']['nb'] = model.get('model_config', {}).get('nb', 8)
            model_opt['network_G']['out_nc'] = model.get('model_config', {}).get('nc_out', 1)
            # CNN do not have network_D param
            if 'network_D' in model_opt:
                model_opt['network_D']['in_nc'] = model.get('model_config', {}).get('nc_in', 1)
                model_opt['network_D']['nf'] = model.get('model_config', {}).get('nf', 64)

            model_opt['train'].update(models_param['train_param'])
        elif mode == 'test':
            model_opt['is_train'] = False
            model_opt['path'] = {'pretrained_G':model['pretrained_G']}
        else:
            raise ValueError('Unknown mode found, Must be train or test!')
        model_opt['network_G']['scale'] = models_param.get('scale', 1)
        model_opt['gpu_id'] = models_param.get('gpu_id', None)
        # Set up parameter for sliding window during inference
        model_opt['dataset_opt'] = data_specific_opt

        # Create model and convert to fp16 for faster computation
        dl_model = create_model(model_opt)
        if mode == 'train':
            # Initialize amp for mixed precision training, determines best precision for each layer
            dl_model.initialize_amp()
        else:
            dl_model.half()

        for dataset in datasets:
            out_d = os.path.join(session_path, current_mod, dataset['name'], mode)
            os.makedirs(out_d, exist_ok=True)

            if mode == 'train':
                model_opt['path']['models'] = os.path.join(out_d , model['name'], 'models')
                model_opt['path']['training_state'] = os.path.join(out_d, model['name'], 'training_state')
                model_opt['path']['tb_logger'] = os.path.join(out_d, model['name'], 'tb_logger')
                # model_opt['path']['val_images'] = os.path.join(out_d, model['name'], 'val_images')
                seed = model_opt['train'].get('manual_seed', None)
                if seed is None:
                    import random
                    seed = random.randint(1, 10000)
                helper_func.set_random_seed(seed)
                helper_func.mkdirs((path for key, path in model_opt['path'].items()))
                
                torch.backends.cudnn.benchmark = True
                from torch.utils.tensorboard import SummaryWriter
                # Summary writer
                tb_logger = SummaryWriter(log_dir=model_opt['path']['tb_logger'])
                # load dataset-specific config
                dataset_opt = config['Datasets'][dataset['name']]
                if dataset.get('in_uids', None):
                    dataset_opt['in_uids'] = dataset['in_uids']
                if dataset.get('tar_uids', None):
                    dataset_opt['tar_uids'] = dataset['tar_uids']
                dataset_opt['is_train'] = model_opt['is_train']
                dataset_opt['out'] = out_d
                dataset_opt['batch_size'] = models_param.get('batch_size', 8)
                dataset_opt['use_shuffle'] = models_param.get('use_shuffle', True)
                dataset_opt['n_workers'] = models_param.get('n_workers', 8)
                dataset_opt.update(data_specific_opt)

                train_set = create_dataset(dataset_opt)
                train_size = int(math.floor(len(train_set) / dataset_opt['batch_size']))
                global_logger.info('Train set {} / Batch size {} (Note: drop_last=True in dataloader!)'.format(len(train_set), dataset_opt['batch_size']))
                total_iters = float(model_opt['train']['niter'])
                total_epochs = int(math.ceil(total_iters / train_size))
                train_loader = create_dataloader(train_set, dataset_opt)

                # Resume training state capability will be implemented soon!
                current_step = 0 
                start_epoch = 0
                ## =====================
                ## START TRAINING
                ## =====================
                for epoch in range(start_epoch, total_epochs):
                    for _, train_data in enumerate(train_loader):
                        current_step += 1
                        if current_step > total_iters:
                            break

                        dl_model.feed_train_data(train_data)
                        dl_model.optimize_parameters(current_step)
                        # Update learning rate, called after .step() after pytorch 1.1
                        dl_model.update_learning_rate()

                        if current_step % model_opt['train']['print_freq'] == 0:
                            logs = dl_model.get_current_log() # Get loss dictionary
                            message = 'Iter:{:8,d}, lr:{:.6e}> '.format(
                                current_step, dl_model.get_current_learning_rate())
                            for k, v in logs.items():
                                message += '{:s}: {:.4e} '.format(k, v)
                                # Log scalars
                                tb_logger.add_scalar(k, v, current_step)
                            global_logger.info(message)

                        # save models and training states at every that step
                        if current_step % model_opt['train']['save_checkpoint_freq'] == 0:
                            global_logger.info('Saving model weights at step {}...'.format(current_step))
                            dl_model.save(current_step)
                            dl_model.save_training_state(epoch, current_step) 
                global_logger.info('End of training!')

            else:
                # load dataset-specific config
                dataset_opt = config['Datasets'][dataset['name']]
                if dataset.get('in_uids', None):
                    dataset_opt['in_uids'] = dataset['in_uids']
                dataset_opt['is_train'] = model_opt['is_train']
                dataset_opt['out'] = out_d
                dataset_opt.update(data_specific_opt)
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
                        helper_func.save_volume(sr_vol, out_type=models_param['out_dtype'], out_dir=os.path.join(out_d, data['uid'][0]), m_type='Volume', f_name='{}--{}'.format(model['name'], data['uid'][0]), target_scale=models_param.get('scale', None))
                        bar()