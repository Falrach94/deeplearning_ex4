
class DataFilter:
    def __init__(self):
        pass

    @staticmethod
    def filter_out_non_labeled_data(data):
        return data[(data.inactive != 0) | (data.crack != 0)]


class AugmentFilter:
    @staticmethod
    def filter_unlabled_augments(df):
        return df[(df.aug == 0) | (df.inactive != 0) | (df.crack != 0)]

