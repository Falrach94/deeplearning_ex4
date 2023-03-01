import os
from math import ceil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from utils.stat_tools import categorize_data


class DataReaderTree:

    def __init__(self):

        self._data = np.load('assets/forest_data.npy', allow_pickle=True)

    def get_csv_data(self, data_descriptor):
        split = data_descriptor.get('valsplit').get_value()
        set = int(data_descriptor.get('set').get_value())

        frame = pd.DataFrame(self._data[set], columns=['path', 'crack', 'inactive'])

        frame_val = pd.DataFrame(self._data[(set+1)%4], columns=['path', 'crack', 'inactive'])

        fine, not_fine = categorize_data(frame_val)

        fine = pd.DataFrame(fine, columns=['path', 'crack', 'inactive'])

        tr, val = train_test_split(frame, test_size=split)

        val = pd.concat((val, fine))
        return tr, val
