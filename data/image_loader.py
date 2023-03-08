import sys
from threading import Lock

import torch
from skimage.color import gray2rgb
from skimage.io import imread


class ImageLoader:
    def __init__(self, image_path_col, rel_path='assets/'):
        self.image_path_col = image_path_col
        self.rel_path = rel_path

    def get(self, df, idx):
        return self._get(df, idx)

    def _get(self, df, idx):
        image_path = df.loc[idx, self.image_path_col]
        image = self._load_image(image_path)
        return image

    def _load_image(self, path):
        image = imread(self.rel_path+path)
        image = torch.tensor(gray2rgb(image)).transpose(0, 2).transpose(1, 2)
        return image


class CachedImageLoader(ImageLoader):
    def __init__(self, image_path_col):
        super().__init__(image_path_col)
        self.image_cache = dict()

    def get(self, df, idx):
        key = self._calc_key(df, idx)
        image = self.image_cache.get(key)

        if image is None:
            image = self._get(df, idx)
            self.image_cache[key] = image

        return image

    @staticmethod
    def _calc_key(self, df, idx):
        return df.loc[idx, self.image_path_col]


class AugmentedImageLoader(CachedImageLoader):
    def __init__(self, image_path_col, augmentor):
        super().__init__(image_path_col)
        self.augmentor = augmentor

    def _calc_key(self, df, idx):
        return (df.loc[idx, self.image_path_col],
                self.augmentor.get_aug_idx(df, idx))

    def _get(self, df, idx):
        image_path, aug_idx = self._calc_key(df, idx)
        if aug_idx == 0:
            base_image = self._load_image(image_path)
        else:
            base_image = self.image_cache.get((image_path, 0))
            if base_image is None:
                base_image = self._load_image(image_path)
                self.image_cache[(image_path, 0)] = base_image

        image = self.augmentor.augment_image(base_image, aug_idx)
        return image



