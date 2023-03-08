import pandas as pd


class CSVReader:

    def __init__(self, path, seperator=','):
        self.data = pd.read_csv(path, sep=seperator)

    def get(self, filter_fct=None):
        if filter_fct is None:
            return self.data
        return filter_fct(self.data)
