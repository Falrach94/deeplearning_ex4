import os
from math import ceil

import numpy as np
import pandas
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from utils.stat_tools import fully_categorize_data


class BigDataReader:

    def __init__(self):
        self._data = pd.read_csv('assets/data.csv', sep=';')
        aug = pd.read_csv('assets/main_data_augs.csv', sep=',')
        extra = pd.read_csv('assets/data_dif.csv', sep=';')
        self._data = pd.concat((self._data, aug, extra))

    def get_csv_data(self, data_descriptor):
        return train_test_split(self._data,
                                test_size=data_descriptor.get('ValSplit').get_value())

