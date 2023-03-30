import torchvision as tv
from torchvision.transforms import InterpolationMode

from data.augment_fuser import BalancedFuser, SimpleFuser, FuserTypes
from data.augment_generator import CustomAugmentor, AugmenterTypes
from data.data_filter import NoAugsFilter, FilterTypes
from data.image_loader import ImageLoaderTypes
from data.label_provider import SimpleLabeler, SingleLabeler, LabelerTypes
from utils.utils import mirror_horizontal, mirror_vertical, rotate90deg, mirror_and_rotate

DATA_PATH = 'assets/data.csv'
#DATA_PATH = 'assets/tr_data.csv'
#DATA_PATH = 'assets/elpv_data.csv'
#DATA_PATH = 'assets/data_seg.csv'
CSV_SEPERATOR = ','


HOLDOUT_SPLIT = 0.15


TR_MEAN = [0.59685254, 0.59685254, 0.59685254]
TR_STD = [0.16043035, 0.16043035, 0.16043035]

TR_TRANSFORMS = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                       tv.transforms.ToTensor(),
 #                                      tv.transforms.GaussianBlur(7),
 #                                      tv.transforms.RandomAutocontrast(),
 #                                      tv.transforms.RandomAdjustSharpness(0.8),
                                      # tv.transforms.RandomSolarize(),
                                       tv.transforms.Normalize(TR_MEAN, TR_STD),
                                       tv.transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
                                       ])
VAL_TRANSFORMS = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                        tv.transforms.ToTensor(),
                                        tv.transforms.Normalize(TR_MEAN, TR_STD),])

IMAGE_LOADER_TYPE = ImageLoaderTypes.Augmented
IMAGE_LOADER_CONFIG = {
    'col': 'filename'
}


LABELER_TYPE = LabelerTypes.SIMPLE
LABEL_COLUMNS = ['crack', 'inactive']
LABELER_OM = 'raw'
LABELER_CONFIG = {
    'cols': LABEL_COLUMNS,
    'om': LABELER_OM
}


FUSER_TYPE = FuserTypes.BALANCED
FUSER_USE_OVERSAMPLING = False
FUSER_CONFIG = {'oversampling': FUSER_USE_OVERSAMPLING}


AUGMENTER_TYPE = AugmenterTypes.CUSTOM
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
AUGMENTATIONS = [
    lambda x: mirror_and_rotate(x, True, False, 0),
    lambda x: mirror_and_rotate(x, False, True, 0),
    lambda x: mirror_and_rotate(x, True, True, 0),  
    lambda x: mirror_and_rotate(x, False, False, 1),
    lambda x: mirror_and_rotate(x, False, False, 2),
    lambda x: mirror_and_rotate(x, False, False, 3),
]
AUGMENTER_CONFIG = {
    'augments': AUGMENTATIONS
}


VAL_FILTER_TYPE = FilterTypes.NO_AUGS
VAL_FILTER_CONFIG = None
TR_FILTER_TYPE = #FilterTypes.NO_AUGS #FilterTypes.SMALL_SET
TR_FILTER_CONFIG = None #{'size': 0.05}
