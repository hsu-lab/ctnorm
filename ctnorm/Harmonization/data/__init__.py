import logging
import os
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


def read_csv(path_to_csv, required_columns=['uids']):
    if isinstance(path_to_csv, str):
        import pandas as pd
        df = pd.read_csv(path_to_csv)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise RuntimeError(f"Error: Missing columns in DataFrame {missing_columns} from file {path_to_csv}")
    else:
        raise NotImplementedError('Unknown uids type found! Must be path to CSV')
    return df


"""
Create Dataset object for dataloader
"""
def create_dataset(dataset_opt):
    from .loader import Loader as L
    if dataset_opt.get('in_uids'):
        in_df = read_csv(dataset_opt['in_uids'], required_columns=['uids'])
        in_uids_list = in_df['uids'].tolist()
        dataset_opt['in_uids_names'] = [in_uid.rstrip() for in_uid in in_uids_list]
        # dataset_opt['in_uids_names'] = sorted(in_uids_list, key=lambda x: os.path.basename(x))
        # Check if mask exists
        if 'masks' in in_df.columns:
            in_masks_list = in_df['masks'].tolist()
            dataset_opt['in_masks_names'] = [in_uid.rstrip() for in_uid in in_masks_list]
        # Check if target exists
        if 'tar_uids' in in_df.columns:
            in_tar_list = in_df['tar_uids'].tolist()
            dataset_opt['in_tar_names'] = [in_uid.rstrip() for in_uid in in_tar_list]
    else:
        raise RuntimeError('CSV with path to input cases must be specified!')
    """
    if dataset_opt.get('tar_uids'):
        tar_uid_list = read_csv(dataset_opt['tar_uids'])['uids'].tolist()
        tar_uid_list = [in_uid.rstrip() for in_uid in tar_uid_list]
        dataset_opt['tar_uids_names'] = sorted(tar_uid_list, key=lambda x: os.path.basename(x))
        # Ensure both in_uids and tar_uids have same path
        compare_in_tar(list1=dataset_opt['in_uids_names'], list2=dataset_opt['tar_uids_names'])
    """
    dataset = L(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('[{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['description']))
    return dataset


def compare_in_tar(list1, list2):
    """
    Compares two lists of file paths in order
    Args:
        list1 (list of str): The first list of file paths.
        list2 (list of str): The second list of file paths.
    Raises:
        ValueError: If the base filenames (in order) do not match.
    Returns:
        None: If the lists match exactly.
    """
    # Extract the base filenames from each list
    basenames1 = [os.path.basename(path) for path in list1]
    basenames2 = [os.path.basename(path) for path in list2]
    if basenames1 != basenames2:
        raise ValueError(
            f"in_uids and tar_uids lists do not match!"
        )