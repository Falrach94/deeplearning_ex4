import shutil

import pandas as pd

from cli_program.settings.data_settings import DATA_PATH
from data.data_filter import OnlyDefectsFilter
from data.dataset_generator import create_dataset
from data.image_loader import ImageLoader
from model.NNModels.ResNet34_pre import ResNet34Sig

data = pd.read_csv('../assets/data.csv')
data_elpv = pd.read_csv('../assets/elpv_data.csv')

data_elpv['nbr'] = data_elpv['filename'].str[-8:-4].astype(int)
data_elpv = data_elpv.set_index('nbr')
data = data.set_index('nbr')
data = data.sort_index()
data_elpv = data_elpv.sort_index()

#data['aux_label'] = data_elpv['inactive']
data_elpv['inactive'] *= 3
data_elpv['inactive'] = round(data_elpv['inactive']).astype(int)

data['aux_label'] = data_elpv['inactive']

ix = [i not in data.index for i in data_elpv.index]
dif = data_elpv[ix].sort_index()

for file in dif.filename:
    shutil.copy2('../assets/'+file, '../assets/dif_images/')

dif.to_csv('../assets/data_ex.csv')
data.to_csv('../assets/data.csv')
'''
data_ix = list(data.index)
elpv_ix = list(data_elpv.index)
ix = [ix in data_ix for ix in elpv_ix]
#ix = [ix for ix in data_ix if ix in elpv_ix]

data = data_elpv[ix, 'inactive']

data[data.index, 'aux_label'] = data_elpv[data.index, 'inactive']

test = 0
'''

