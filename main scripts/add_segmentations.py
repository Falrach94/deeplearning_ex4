import os

import numpy as np
import pandas as pd
from skimage.io import imsave

from cli_program.settings.data_settings import DATA_PATH, CSV_SEPERATOR, LABEL_PROVIDER, TR_TRANSFORMS, VAL_TRANSFORMS
from data.augment_fuser import SimpleFuser, RejectAugmentsFuser
from data.augment_generator import CustomAugmentor
from data.data_reader import CSVReader
from data.dataset_generator import create_dataset
from data.image_loader import AugmentedImageLoader
from model.NNModels.AutoEncoder import ResNetAutoEncoder
from model.NNModels.autoenc.SegmentationModel import SegmentationModel
from utils.console_util import print_progress_bar, ScreenBuilder

REL_PATH = 'seg_images/'
IMAGE_OUTPUT_PATH = "assets/" + REL_PATH
CSV_OUTPUT_PATH = "assets/data_seg.csv"

model = SegmentationModel()

fuser = RejectAugmentsFuser()

augmentor = CustomAugmentor(fuser, [])
image_provider = AugmentedImageLoader(image_path_col='filename',
                                      augmentor=augmentor)
df_org = CSVReader(path=DATA_PATH, seperator=CSV_SEPERATOR).get()
df = LABEL_PROVIDER.label_dataframe(df_org)
data = create_dataset(df, image_provider, LABEL_PROVIDER, augmentor, VAL_TRANSFORMS, 16, None, False)

LABEL_PROVIDER.set_output_mode('name')

model.cuda()
model.eval()

MEAN = 0.59685254
STD = 0.16043035

sb = ScreenBuilder()

if not os.path.isdir(IMAGE_OUTPUT_PATH):
    os.mkdir(IMAGE_OUTPUT_PATH)

t = df_org['filename'].str[-12:]

df_org['filename'] = REL_PATH + df_org['filename'].str[-12:]
df_org.to_csv(CSV_OUTPUT_PATH, index=False)

print_progress_bar('progress', 0, len(data['dl']), '', sb=sb)
for i, (x, y) in enumerate(data['dl']):
    x = x.cuda()
    images = np.array(model(x).cpu())
    images = images.transpose((0, 2, 3, 1))
    images *= STD
    images += MEAN
    print_progress_bar('progress', i, len(data['dl']), '', sb=sb)


    for im, path in zip(images, y):
        path = IMAGE_OUTPUT_PATH + path[-12:]
        imsave(path, np.round(255*im).astype(np.uint8))

