import os
from math import ceil

import numpy as np
import pandas
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from utils.stat_tools import fully_categorize_data


class SmallDataReader:

    def __init__(self, prune=True, split=0.2):
        data = pd.read_csv('assets/data.csv', sep=',')
        aug = pd.read_csv('assets/main_data_augs.csv', sep=',')

        self.tr_data, self.val_data = train_test_split(data,
                                                       test_size=split)

        val_ix = self.val_data.set_index('nbr').index
        aug_ix = aug.set_index('nbr').index
        selection = ~aug_ix.isin(val_ix)
        aug = aug[selection]

        if prune:
            aug = aug.drop(aug[(aug.crack == 0) & (aug.inactive == 0)].index)

        self.tr_data = pd.concat((self.tr_data, aug))


    def get_csv_data(self, data_descriptor):
        return self.tr_data, self.val_data

