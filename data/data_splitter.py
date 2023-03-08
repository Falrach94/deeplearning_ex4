import numpy as np
import pandas as pd


def k_fold_split(data, k):
    ix = np.arange(len(data))
    np.random.shuffle(ix)
    data = data.iloc[ix].reset_index()

    splits = [(int(len(data)*i/k),
               int(len(data)*(i+1)/k)) for i in range(k)]

    folds = [{
        'tr': pd.concat((data[:i], data[j:])),
        'val': data[i:j]
    } for i, j in splits]

    return folds
