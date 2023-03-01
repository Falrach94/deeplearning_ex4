import os
from math import ceil

import numpy as np
import pandas
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from utils.stat_tools import fully_categorize_data


class InactiveReader:

    def __init__(self):

        # load the reader from the csv file and perform a train-test-split
        # this can be accomplished using the already imported pandas and sklearn.model_selection modules
        self._data = pd.read_csv('assets/data2.csv', sep=',')
        aug = pd.read_csv('assets/data_augs.csv', sep=',')
        self._data = pd.concat((self._data, aug))

    def get_csv_data(self, data_descriptor):
        return train_test_split(self._data, test_size=data_descriptor.get('ValSplit').get_value())
