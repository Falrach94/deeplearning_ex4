import numpy as np
import pandas as pd


def k_fold_split(data, k):
    ix = np.arange(len(data))
    np.random.shuffle(ix)
    data = data.iloc[ix].reset_index(drop=True)

    splits = [(int(len(data)*i/k),
               int(len(data)*(i+1)/k)) for i in range(k)]

    folds = [{
        'tr': pd.concat((data[:i], data[j:])).reset_index(drop=True),
        'val': data[i:j].reset_index(drop=True)
    } for i, j in splits]

    return folds
