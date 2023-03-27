from data.utils import get_distribution, split_df_by_category


class DataFilter:
    def __init__(self):
        pass

    @staticmethod
    def filter_out_non_labeled_data(data):
        return data[(data.inactive != 0) | (data.crack != 0)]


class AugmentFilter:
    @staticmethod
    def filter_unlabled_augments(df, old_df):
        return df[(df.aug == 0) | (df.inactive != 0) | (df.crack != 0)].copy().reset_index()

class NoDefectsFilter:
    @staticmethod
    def filter(df):
        df_cat = split_df_by_category(df)
        return df_cat[0].reset_index(drop=True)


class SmallSetFilter:
    def __init__(self, size):
        self.size = size

    def filter(self, df):
        return df.sample(frac=self.size).reset_index(drop=True)

class OnlyDefectsFilter:
    @staticmethod
    def filter(df):
        return df[(df.inactive != 0) | (df.crack != 0)].sample(frac=1).reset_index(drop=True)
