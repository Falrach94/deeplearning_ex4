import torch
import torchvision as tv

from utils.utils import mirror_horizontal, mirror_vertical, rotate90deg

DATA_PATH = 'assets/data.csv'
CSV_SEPERATOR = ','

LABEL_COLUMNS = ['crack', 'inactive']

HOLDOUT_SPLIT = 0.2

AUGMENTATIONS = [
    mirror_vertical,
    mirror_horizontal,
    lambda x: rotate90deg(x, 1),
    lambda x: rotate90deg(x, 2),  # 180 deg
    lambda x: rotate90deg(x, 3),  # 270 deg
]

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
