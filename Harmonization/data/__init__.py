import logging
import torch.utils.data


"""
Create dataloader given the dataset object and other parameter dictionary
"""
def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['mode']
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['n_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1)


"""
Create Dataset object for dataloader
"""
def create_dataset(dataset_opt, uids=False):
    from .loader import Loader as L
    if dataset_opt.get('uids'):
        if isinstance(dataset_opt.get('uids'), str):
            with open(dataset_opt.get('uids'), 'r') as f:
                lines = f.readlines()
            dataset_opt['uids'] = [l.rstrip() for l in lines]
        elif isinstance(dataset_opt.get('uids'), list):
            pass
        else:
            raise NotImplementedError('Unknown uids type found!')
    else:
        dataset_opt['uids'] =  os.listdir(dataset_opt['input_path'])
    dataset = L(dataset_opt) # create dataobject
    logger = logging.getLogger('base')
    logger.info('[{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['description']))
    return dataset