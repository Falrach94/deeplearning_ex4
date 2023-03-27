import pandas as pd
from sklearn.model_selection import train_test_split

from cli_program.settings.data_settings import DATA_PATH

data = pd.read_csv(DATA_PATH)

SPLIT = 0.1

tr, val = train_test_split(data, test_size=SPLIT)

tr.to_csv('assets/tr_data.csv', index=False)
val.to_csv('assets/val_data.csv', index=False)
