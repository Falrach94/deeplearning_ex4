from torch.utils.data import Dataset
import torch
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv


train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class AutoencoderDataset(Dataset):

    def autoencode(self):
        self._autoencode = True

    def label(self):
        self._autoencode = False

    def __init__(self, data, mode, transform_mode=0, normalize=True):
        super().__init__()

        self._autoencode = False

        self.train = (mode == "train")
        self._data = data

        if transform_mode != 0:
            transform_mode = 1

        if normalize:
            transforms = [tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                 tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize(train_mean, train_std)]),
                          tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                 tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize(train_mean, train_std),
                                                 tv.transforms.GaussianBlur(7),
                                                 tv.transforms.RandomAutocontrast()])]
        else:
            transforms = [tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                 tv.transforms.ToTensor()]),
                          tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                 tv.transforms.GaussianBlur(7),
                                                 tv.transforms.RandomAutocontrast(),
                                                 tv.transforms.ToTensor()])]
                                                 #tv.transforms.GaussianBlur(7),
                                                 #tv.transforms.RandomAutocontrast()])]

        if self.train:
            self._transform = transforms[transform_mode]
        else:
            self._transform = transforms[0]

        self.images = [None]*len(self._data)

        self._data.reset_index(inplace=True)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.images[idx] is None:
            rel_path = f'assets/{self._data.loc[idx, "filename"]}'
            image = torch.tensor(imread(rel_path))
            image = gray2rgb(image)
            self.images[idx] = image

        image = self._transform(self.images[idx])

        if self._autoencode:
            return image, image
        else:
            cracked = self._data.loc[idx, "crack"]
            inactive = self._data.loc[idx, "inactive"]
            return image, torch.tensor([float(cracked), float(inactive)])
