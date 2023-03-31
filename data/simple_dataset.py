import torch
from torch.utils.data import Dataset

from data.utils import get_distribution


class SimpleDataset(Dataset):

    def __init__(self, data, image_provider, label_provider, transforms):
        self._data = data
        self.label_provider = label_provider
        self.image_provider = image_provider
        self.transforms = transforms

        self.add_idx = False

    def set_add_idx(self, add_idx):
        self.add_idx = add_idx

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):

        image = self.image_provider.get(self._data, idx)
        image = self.transforms(image)
        label = self.label_provider.get_label(self._data, idx, image)

        if self.add_idx:
            return image, (label, self._data.loc[idx, 'nbr'])
        return image, label

    def get_categories(self):
        return get_distribution(self._data, self.label_provider)
