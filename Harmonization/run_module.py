import os
from .data import create_dataset, create_dataloader


def main(config, global_logger, session_path):
    current_mod = os.path.basename(os.path.dirname(__file__))
    datasets = config[current_mod]['input_datasets']
    models = config[current_mod]['models']
    mode = config[current_mod]['mode']
    if mode == 'test':
        for dataset in datasets:
            print('dataset:', dataset)
            out_dir = os.path.join(session_path, current_mod, dataset)
            os.makedirs(out_dir, exist_ok=True)
            dataset_opt = config['Datasets'][dataset]
            dataset_opt['mode'] = mode
            test_set = create_dataset(dataset_opt)
            test_loader = create_dataloader(test_set, dataset_opt)
            # data = iter(test_loader).next()
            # print(data.keys())
    elif mode == 'train':
        pass
    
    else:
        global_logger.info(f'Mode {mode} not implemented!')