import os
import uuid
from math import ceil

import torchvision as tv
import pandas as pd
import numpy as np
import torch
from imageio import imwrite
from skimage.color import gray2rgb
from skimage.io import imread
from sklearn.model_selection import train_test_split

from model.NNModels.ResNet50v2_pre import ResNet50v2_Pretrained
from utils.stat_tools import categorize_data

'''
data_a = pd.read_csv('assets/data.csv', sep=';')
data_a['nbr'] = data_a['filename'].str[-8:-4]
data_a.sort_values('nbr', inplace=True)

data_b = pd.read_csv('assets/data2.csv', sep=',')
data_b['nbr'] = data_b['path'].str[-8:-4]
data_b.sort_values('nbr', inplace=True)

data_b.rename(columns={'path': 'filename'}, inplace=True)


data = pd.concat((data_a, data_b)).drop_duplicates(subset='nbr', keep=False)
data.drop('nbr', axis=1, inplace=True)
data.reset_index(inplace=True)

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
model = ResNet50v2_Pretrained()

#def pred(path):
#    image = image[np.newaxis, ...]
#    pred = model.forward(image)
#    return pred

cp_path = "assets/good_models/787_791_50v2.ckp"
cp = torch.load(cp_path, 'cuda')
model.load_state_dict(cp['state_dict'])

#data = pd.read_csv('assets/data_dif.csv', sep=';')

images = np.zeros((len(data), 3, 300, 300))

transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                   tv.transforms.ToTensor(),
                                   tv.transforms.Normalize(train_mean, train_std)])
for i, row in data.iterrows():
    image = np.array(imread('assets/' + row['filename']))
    image = gray2rgb(image)
    image = transform(image)
    images[i, ...] = image

model.cuda()

pred = torch.empty((images.shape[0], 2))
images = torch.tensor(images, dtype=torch.float)
num = images.shape[0]
cnt = int(images.shape[0]/64+0.5)
model.requires_grad_(False)
for i in range(cnt):
    print(i+1, '/', cnt)
    start = i*64
    end = (i+1)*64
    if end > num:
        end = num
    batch = images[start:end]
    batch = batch.cuda()
    y = model(batch)
    pred[start:end] = y.cpu()
    #del batch

data = np.array(data)[:, 1:]
data = pd.DataFrame(data, columns=['filename', 'crack', 'inactive'])

for i in range(len(data)):
    oi = data['inactive'][i]
    cracked = pred[i][0].item()
    val = 1 if data['inactive'][i] > 0.5 else 0
    data.at[i, 'inactive'] = val
    if data.at[i, 'inactive'] == 1:
        if cracked > 0.5:
            data.at[i, 'crack'] = 1
        elif cracked > 0.1:
            data.at[i, 'crack'] = 0.5
        else:
            data.at[i, 'crack'] = 0
    else:
        if cracked > 0.75:
            data.at[i, 'crack'] = 1
        if cracked > 0.35:
            data.at[i, 'crack'] = 0.5
        else:
            data.at[i, 'crack'] = 0
    print(i, 'of', len(data), f'inactive: ({oi}, {data.at[i, "inactive"]}); cracked: ({round(cracked,2)}, {data.at[i, "crack"]})')

data.to_csv('assets/data_dif.csv', sep=';', index=False)
'''


data2 = pd.read_csv('../assets/data_dif.csv', sep=';')
#data2 = np.concatenate((data2, np.empty((data2.shape[0], 1))), axis=1)
#data2[:, 2] = data2[:, 1]
#data2[:, 1] = 0.5
#data2 = pd.DataFrame(data2, columns=['path', 'crack', 'inactive'])
#data2.to_csv('assets/data2.csv', sep=';')


new_images = []

def make_image(image, entry, mod):
    name = entry['filename'][-12:-4]
    path = "images/augs/" + name + "_" + mod + ".png"
    imwrite('assets/' + path, image)
    new_images.append((path, entry['crack'], entry['inactive']))

for i, entry in data2.iterrows():
#    if entry['inactive'] > 0.5 or entry['crack'] > 0.5:
    image = torch.tensor(imread('assets/' + entry['filename']))

    make_image(torch.fliplr(image), entry, 'vflip')
    make_image(torch.flipud(image), entry, 'hflip')
    image = torch.rot90(image, dims=(0,1))
    make_image(image, entry, '90deg')
    image = torch.rot90(image, dims=(0,1))
    make_image(image, entry, '180deg')
    image = torch.rot90(image, dims=(0,1))
    make_image(image, entry, '270deg')
    print(str(i) + "/" + str(len(data2)))

#augs = pd.read_csv('assets/data_augs.csv', sep=';')
data = np.array(new_images)
data = pd.DataFrame(data, columns=['filename', 'crack', 'inactive'])
#data = pd.concat((augs, data))
data.to_csv('assets/big_data_augs.csv', sep=';', index=False)
'''
data = pd.read_csv('assets/data.csv', sep=';')

label = np.array(data)[:, 1:3]

s = np.sum(label, axis=0)

just_cracked = 0
just_inactive = 0
both = 0
fine = 0

for l in label:
    if l[0] + l[1] == 2:
        both += 1
    elif l[0] == 1:
        just_cracked += 1
    elif l[1] == 1:
        just_inactive += 1
    else:
        fine += 1

print('total:', label.shape[0])
print('cracks:', s[0], 'defects:', s[1])
print('fine', fine, 'just crack', just_cracked, 'just inactive', just_inactive, 'both', both)


def split_data(data, cnt):
    total_cnt = len(data)
    split_cnt = ceil(total_cnt / cnt)

    res = np.empty((split_cnt, cnt, 3), dtype=object)

    index = 0
    for i in range(split_cnt):
        if index + cnt <= total_cnt:
            res[i, :] = data[index:index + cnt]
            index += cnt
        else:
            remaining = total_cnt - index
            res[i, :remaining] = data[index:]
            res[i, remaining:cnt] = data[cnt - remaining]
    return res, split_cnt
'''
'''
# load the reader from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
csv_path = ''
for root, _, files in os.walk('assets'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)
data = pd.read_csv(csv_path, sep=';')

fine, not_fine = categorize_data(data)
fine, cnt = split_data(fine, len(not_fine))

not_fine = np.array(not_fine)[np.newaxis, ...]
not_fine = np.tile(not_fine, (cnt, 1, 1))
data = np.concatenate((fine, not_fine), axis=1)
idx = torch.randperm(data.shape[1])
data = data[:, idx, :]  # .reshape(tr_data.shape[1], tr_data.shape[0], tr_data.shape[2])

np.save('assets/forest_data', data, allow_pickle=True)
'''