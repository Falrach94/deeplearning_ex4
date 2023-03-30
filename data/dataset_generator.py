from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from data.data_splitter import k_fold_split
from data.simple_dataset import SimpleDataset
from model.config import WORKER_THREADS

def _create_dataset(df,
                    image_provider,
                    label_provider,
                    augmenter,
                    transform,
                    batch_size,
                    filter,
                    shuffle=True):

    if augmenter is not None:
        df = augmenter.add_augmentations_to_df(df)
    if filter is not None:
        df = filter.filter(df)


    dataset = SimpleDataset(df,
                           transforms=transform,
                           image_provider=image_provider,
                           label_provider=label_provider)
    dl = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=WORKER_THREADS,
                    persistent_workers=True)

    return {'dataset': dataset, 'dl': dl}

def _create_datasets(tr_data, val_data,
                     image_provider, label_provider,
                     augmentor,
                     tr_transform, val_transform,
                     batch_size,
                     tr_filter, val_filter):

    tr = _create_dataset(tr_data,
                         image_provider,
                         label_provider,
                         augmentor,
                         tr_transform,
                         batch_size,
                         tr_filter,
                         True)

    val = _create_dataset(val_data,
                         image_provider,
                         label_provider,
                         augmentor,
                         val_transform,
                         batch_size,
                         val_filter,
                         False)

    return {
        'tr': tr,
        'val': val,
    }


def create_dataset(data, image_provider, label_provider, augmenter, transform, batch_size, filter, shuffle=False):
    return _create_dataset(data, image_provider, label_provider, augmenter, transform, batch_size, filter, shuffle)


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
                           batch_size,
                           tr_filter, val_filter):
    return [_create_datasets(fold['tr'], fold['val'],
                             image_provider, label_provider,
                             augmentor,
                             tr_transform, val_transform,
                             batch_size,
                             tr_filter, val_filter)
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



