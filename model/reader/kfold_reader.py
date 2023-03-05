import os
from math import ceil

import numpy as np
import pandas
import pandas as pd
import sklearn.model_selection
import torch
from sklearn.model_selection import train_test_split

from utils.stat_tools import fully_categorize_data


# creates k fold cross validation sets from data.csv
# augments of labeled elements are added to th  training data sets
class KFoldReader:

    def __init__(self, k=5):
        data = pd.read_csv('assets/data.csv', sep=',').set_index('nbr')
        aug = pd.read_csv('assets/main_data_augs.csv', sep=',').set_index('nbr')

        # remove augmentations of images without labels
        aug = aug[(aug.inactive != 0) | (aug.crack != 0)]

        # split not augmented data into k folds (tr, val)
        kf = sklearn.model_selection.KFold(5, shuffle=True)
        ix = data.index
        sp = kf.split(ix)
        folds_ix = next(sp)
        folds = [(data.iloc[ix[0]], data.iloc[ix[1]]) for ix in folds_ix]

        # for each fold select only augmentations of images not in the validation set
        val_ix = [fold[1].index for fold in folds]
        selection = [~aug.index.isin(ix) for ix in val_ix]
        fold_augs = [aug[sel] for sel in selection]

        # combine folds and augmentations
        self.folds = [(pd.concat((fold[0], augs)), fold[1]) for fold, augs in zip(folds, fold_augs)]

    def get_csv_data(self, data_descriptor):
        return self.tr_data, self.val_data

