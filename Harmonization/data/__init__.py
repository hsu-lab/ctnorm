import logging
import torch.utils.data


"""
Create dataloader given the dataset object and other parameter dictionary
"""
def create_dataloader(dataset, dataset_opt):
    is_train = dataset_opt['is_train']
    if is_train:
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
            import pandas as pd
            df = pd.read_csv(dataset_opt.get('uids'))
            lines = df['uids'].to_list()
            dataset_opt['uids'] = [l.rstrip() for l in lines]
        else:
            raise NotImplementedError('Unknown uids type found!')
    else:
        raise ValueError('CSV with path to cases must be specified!')
    dataset = L(dataset_opt) # create dataobject
    logger = logging.getLogger('base')
    logger.info('[{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['description']))
    return dataset