import pandas as pd

from data.utils import get_distribution, split_df_by_category


class SimpleFuser:
    @staticmethod
    def fuse(df, df_augs):
        return pd.concat(df, df_augs)

class BalancedFuser:

    def __init__(self, label_provider, target_per_category):
        self.label_provider = label_provider

    def fuse(self, df, df_augs):
        distribution = get_distribution(df, self.label_provider)
        max_cnt = max(distribution)

        df_cat = split_df_by_category(df)
        df_augs_cat = split_df_by_category(df_augs)

        df_cat_cnt = [len(frame) for frame in df_cat]
        df_augs_cnt = [len(frame) for frame in df_augs_cat]
        df_augs_cnt = [min(max_cnt - df_cnt, df_a_cnt) for df_cnt, df_a_cnt in zip(df_cat_cnt, df_augs_cnt)]
        df_augs_sel = [frame.sample(cnt) for frame, cnt in zip(df_augs_cat, df_augs_cnt)]
       # return df.sample(200).reset_index()
        return pd.concat((df, *df_augs_sel)).reset_index()
