import os
import pickle
from .data import create_dataset, create_dataloader
from .models import create_model


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
            model_opt['is_train'] = False
            model_opt['network_G']['scale'] = models_param['scale']
            model_opt['gpu_id'] = models_param['gpu_id']
            model_opt['path'] = {'pretrain_model_G':models[model]}
            model_opt['dataset_opt'] = {
                'tile_xy': models_param['tile_xy'] if models_param['tile_xy'] is not None else 512,
                'tile_z': models_param['tile_z'] if models_param['tile_z'] is not None else 32,
                'z_overlap': models_param['z_overlap'] if models_param['z_overlap'] is not None else 4,
                'scale': models_param['scale'] if models_param['scale'] is not None else 1.0
            }
            # create model and convert to fp16 for faster computation
            model = create_model(model_opt)
            model.half()

            for dataset in datasets:
                print('dataset:', dataset)
                out_dir = os.path.join(session_path, current_mod, dataset)
                os.makedirs(out_dir, exist_ok=True)
                # load dataset-specific config
                dataset_opt = config['Datasets'][dataset]
                dataset_opt['mode'] = mode
                test_set = create_dataset(dataset_opt)
                test_loader = create_dataloader(test_set, dataset_opt)
                """
                data = iter(test_loader).next()
                print(data.keys())
                """
                for i, data in enumerate(test_loader):
                    need_HR = False if test_loader.dataset.opt.get('dataroot_HR') is None else True
                    print('Inference for case:', data['uid'][0])
                    model.feed_test_data(data, need_HR=need_HR)
                    model.test(data)
                    print('out done!')
                    raise ValueError('Break')

    elif mode == 'train':
        pass
    
    else:
        global_logger.info(f'Mode {mode} not implemented!')