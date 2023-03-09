import torch
import torchvision as tv

from data.data_filter import AugmentFilter
from utils.utils import mirror_horizontal, mirror_vertical, rotate90deg, mirror_and_rotate

DATA_PATH = 'assets/data.csv'
#DATA_PATH = 'assets/elpv_data.csv'

CSV_SEPERATOR = ','

#LABEL_COLUMNS = ['inactive']
LABEL_COLUMNS = ['crack', 'inactive']

HOLDOUT_SPLIT = 0.2

AUGMENTATIONS = [
    lambda x: mirror_and_rotate(x, False, False, 1),
    lambda x: mirror_and_rotate(x, False, False, 2),
    lambda x: mirror_and_rotate(x, False, False, 3),
    lambda x: mirror_and_rotate(x, True, False, 0),
    lambda x: mirror_and_rotate(x, True, False, 1),
    lambda x: mirror_and_rotate(x, True, False, 2),
    lambda x: mirror_and_rotate(x, True, False, 3),
    lambda x: mirror_and_rotate(x, False, True, 0),
    lambda x: mirror_and_rotate(x, False, True, 1),
    lambda x: mirror_and_rotate(x, False, True, 2),
    lambda x: mirror_and_rotate(x, False, True, 3),
]

#AUGMENTATIONS = [
#    mirror_vertical,
#    mirror_horizontal,
#    lambda x: rotate90deg(x, 1),
#    lambda x: rotate90deg(x, 2),  # 180 deg
#    lambda x: rotate90deg(x, 3),  # 270 deg
#]

AUGMENTATION_FILTER = AugmentFilter.filter_unlabled_augments

TR_MEAN = [0.59685254, 0.59685254, 0.59685254]
TR_STD = [0.16043035, 0.16043035, 0.16043035]

TR_TRANSFORMS = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                       tv.transforms.ToTensor(),
                                       tv.transforms.GaussianBlur(7),
                                       tv.transforms.RandomAutocontrast(),
                                       tv.transforms.Normalize(TR_MEAN, TR_STD),])
VAL_TRANSFORMS = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                        tv.transforms.ToTensor(),
                                        tv.transforms.Normalize(TR_MEAN, TR_STD),])
