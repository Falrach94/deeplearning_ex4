from torch.utils.data import Dataset


class SimpleDataset(Dataset):

    def __init__(self, data, image_provider, label_provider, transforms):
        self._data = data
        self.label_provider = label_provider
        self.image_provider = image_provider
        self.transforms = transforms

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        image = self.image_provider.get(self._data, idx)
        if image.size(0) != 3:
            b = 0
        image = self.transforms(image)

        label = self.label_provider.get_label(self._data, idx)

        return image, label

