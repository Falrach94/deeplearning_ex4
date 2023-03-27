import pandas as pd

from data.utils import get_distribution, split_df_by_category


class SimpleFuser:
    @staticmethod
    def fuse(df, df_augs):
        return pd.concat((df, df_augs)).sample(frac=1).reset_index(drop=True)

class RejectAugmentsFuser:
    @staticmethod
    def fuse(df, df_augs):
        return df.sample(frac=1).reset_index(drop=True)

class BalancedFuser:

    def __init__(self, label_provider, target_per_category, oversample=True):
        self.label_provider = label_provider
        self.oversample = oversample
    def fuse(self, df, df_augs):
        distribution = get_distribution(df, self.label_provider)
        target_cnt = max(distribution)

        #split original samples into categories
        df_cat = split_df_by_category(df)

        #split augmented samples into categories
        df_augs_cat = split_df_by_category(df_augs)

        #select augmentations to balance categories as much as possible
        df_cat_cnt = [len(frame) for frame in df_cat]
        df_augs_cnt = [len(frame) for frame in df_augs_cat]
        df_augs_cnt = [min(target_cnt - df_cnt, df_a_cnt) for df_cnt, df_a_cnt in zip(df_cat_cnt, df_augs_cnt)]
        df_augs_sel = [frame.sample(cnt) for frame, cnt in zip(df_augs_cat, df_augs_cnt)]
        df = pd.concat((df, *df_augs_sel))

        #oversample underrepresented classes
        if self.oversample:
            df_cat = split_df_by_category(df)
            remaining_cnt = [0 if len(frame) == 0 else target_cnt - len(frame) for frame in df_cat]
            df_os = [df_c.sample(cnt, replace=True) for cnt, df_c in zip(remaining_cnt, df_cat)]
            df_os = [frame for frame in df_os if len(frame) != 0]
            df = pd.concat((df, *df_os))

        df = df.sample(frac=1).reset_index(drop=True)

        return df
