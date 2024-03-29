import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_distribution(df, label_provider):
    om = label_provider.get_output_mode()
    label_provider.set_output_mode('idx')

    label_series = df.apply(
        lambda row: label_provider.get_label_from_row(row),
        axis=1
    )

    categories = label_series.value_counts()
    for i in range(label_provider.class_count(False)):
        if i not in categories.index:
            categories[i] = 0

    categories = list(categories.sort_index())
    label_provider.output_mode = om
    return categories


def split_df_by_category(df):
    id_col = 'label_id'
    cat_cnt = max(df[id_col])+1
    return [df[df[id_col]==i].reset_index(drop=True) for i in range(cat_cnt)]


def as_list(v):
    return v if isinstance(v, list) else [v]


def combine(list):
    res = [[]]
    for l in list:
        right = res
        res = []
        for y in l:
            for x in right:
                res.append(as_list(x) + as_list(y))
    return res


def split_with_equal_categories(df, labeler, split):
    cnt = labeler.class_count(raw=False)

    part_df = [df[df.label_id == i] for i in range(cnt)]
    splits = np.array([train_test_split(pdf, test_size=split) for pdf in part_df], dtype=object)
    tr_dfs = splits[:, 0]
    val_dfs = splits[:, 1]

    return pd.concat(tr_dfs).reset_index(drop=True),\
        pd.concat(val_dfs).reset_index(drop=True)



