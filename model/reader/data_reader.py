import os
from math import ceil

import numpy as np
import pandas
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from utils.stat_tools import fully_categorize_data


class DataReader:

    def __init__(self):

        # load the reader from the csv file and perform a train-test-split
        # this can be accomplished using the already imported pandas and sklearn.model_selection modules
        self._data = pd.read_csv('assets/data.csv', sep=';')
        aug = pd.read_csv('assets/main_data_augs.csv', sep=',')
        self._data = pd.concat((self._data, aug))

        self._categorized_data = [[],[],[],[]]

        self.oversampling = [1, 4, 10, 5]

#        for l in self.label:
#            if l[1] + l[2] == 2:
#                self._categorized_data[3].append(l)
#            elif l[1] == 1:
#                self._categorized_data[1].append(l)
#            elif l[2] == 1:
#                self._categorized_data[2].append(l)
#            else:
#                self._categorized_data[0].append(l)

    @staticmethod
    def categorize_data(data):
        fine = []
        not_fine = []
        data = np.array(data)[:, 0:3]
        for l in data:
            if l[1] == 1 or l[2] == 1:
                not_fine.append(l)
            else:
                fine.append(l)
        return fine, not_fine

    @staticmethod
    def split_data(data, cnt):
        total_cnt = len(data)
        split_cnt = ceil(total_cnt/cnt)

        res = np.empty((split_cnt, cnt, 3), dtype=object)

        index = 0
        for i in range(split_cnt):
            if index + cnt <= total_cnt:
                res[i, :] = data[index:index+cnt]
                index += cnt
            else:
                remaining = total_cnt-index
                res[i, :remaining] = data[index:]
                res[i, remaining:cnt] = data[cnt-remaining]
        return res, split_cnt

    def get_csv_data(self, data_descriptor):
        return train_test_split(self._data, test_size=data_descriptor.get('ValSplit').get_value())

        split = data_descriptor.hyperparams[1].get_value()
        oversample = int(data_descriptor.hyperparams[3].get_value()) == 1
        tree_cnt = int(data_descriptor.hyperparams[4].get_value())
        forest = tree_cnt > 0

        if forest:
            tr_data, val_data = train_test_split(self._data, test_size=split)
            fine, not_fine = self.categorize_data(tr_data)
            fine, cnt = self.split_data(fine, len(not_fine))
            not_fine = np.array(not_fine)[np.newaxis, ...]
            not_fine = np.tile(not_fine, (cnt, 1, 1))
            tr_data = np.concatenate((fine, not_fine), axis=1)
            idx = torch.randperm(tr_data.shape[1])
            tr_data = tr_data[:, idx, :].reshape(tr_data.shape[1], tr_data.shape[0], tr_data.shape[2])
            return tr_data, val_data

        # oversample
        if oversample:
            tr_data, val_data = train_test_split(self._data, test_size=split)
            cat = fully_categorize_data(tr_data)

            data = None
            for i_cat in range(4):
                for _ in range(self.oversampling[i_cat]):
                    if data is None:
                        data = cat[i_cat]
                    else:
                        data = np.concatenate((data, cat[i_cat]))
            idx = torch.randperm(data.shape[0])
            data = pd.DataFrame(data[idx, :], columns=['filename', 'crack', 'inactive'])
            return data, val_data

        return train_test_split(self._data, test_size=split)
