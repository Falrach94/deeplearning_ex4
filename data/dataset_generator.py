from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from data.data_splitter import k_fold_split
from data.simple_dataset import SimpleDataset
from model.config import WORKER_THREADS


def _create_datasets(tr_data, val_data,
                     image_provider, label_provider,
                     augmentor,
                     tr_transform, val_transform,
                     batch_size,
                     tr_filter, val_filter):

    tr_df = augmentor.add_augmentations_to_df(tr_data)
    if tr_filter is not None:
        tr_df = tr_filter.filter(tr_df)

    val_df = augmentor.add_augmentations_to_df(val_data)
    if val_filter is not None:
        val_df = tr_filter.filter(val_df)

    tr_dataset = SimpleDataset(tr_df,
                               transforms=tr_transform,
                               image_provider=image_provider,
                               label_provider=label_provider)
    tr_dl = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=WORKER_THREADS)
    val_dataset = SimpleDataset(val_df,
                                transforms=val_transform,
                                image_provider=image_provider,
                                label_provider=label_provider)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=WORKER_THREADS)

    return {
        'tr': {'dataset': tr_dataset, 'dl': tr_dl},
        'val': {'dataset': val_dataset, 'dl': val_dl},
    }


def create_single_split_datasets(data, split,
                                 image_provider, label_provider,
                                 augmentor,
                                 tr_transform, val_transform,
                                 batch_size,
                                 tr_filter, val_filter):
    tr_data, val_data = train_test_split(data, test_size=split)
    return _create_datasets(tr_data, val_data,
                            image_provider, label_provider,
                            augmentor,
                            tr_transform, val_transform,
                            batch_size,
                            tr_filter, val_filter)


def create_k_fold_datasets(data, k,
                           image_provider, label_provider,
                           augmentor,
                           tr_transform, val_transform,
                           batch_size):
    return [_create_datasets(fold['tr'], fold['val'],
                             image_provider, label_provider,
                             augmentor,
                             tr_transform, val_transform,
                             batch_size)
            for fold in k_fold_split(data, k)]


def create_k_fold_datasets_with_holdout(data, k, holdout,
                                        image_provider, label_provider,
                                        augmentor,
                                        tr_transform, val_transform,
                                        batch_size):
    if holdout is not None:
        data, holdout_df = train_test_split(data, test_size=holdout)
        holdout_sets = _create_datasets(data, holdout_df,
                                        image_provider, label_provider,
                                        augmentor,
                                        tr_transform, val_transform,
                                        batch_size)
    else:
        holdout_sets = None

    return {
        'holdout': holdout_sets,
        'folds': create_k_fold_datasets(data, k,
                                        image_provider, label_provider,
                                        augmentor,
                                        tr_transform, val_transform,
                                        batch_size)
        }



