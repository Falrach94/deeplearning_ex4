import pandas as pd
import torch


DEFAULT_AUGMENTS = [
    torch.fliplr,
    torch.flipud,
    torch.rot90,
    lambda im: torch.rot90(im, k=2),  # 180 deg
    lambda im: torch.rot90(im, k=3),  # 270 deg
]


class BaseAugmentor:
    def __init__(self, col_name, augment_cnt, fuser):
        self.cnt = augment_cnt
        self.col_name = col_name
        self.fuser = fuser

    '''
    - adds an additional column containing the augmentation to be used
    - adds a copy of each row for each possible augmentation
    '''
    def add_augmentations_to_df(self, df, identity=False):
        df = df.copy()
        df[self.col_name] = 0

        if identity or self.cnt == 0:
            return df

        #create dataframe with augments
        augmented_df = [df.copy() for _ in range(self.cnt)]
        for i, df_aug in enumerate(augmented_df):
            df_aug[self.col_name] = i
        augmented_df = pd.concat(augmented_df).reset_index()

        #fuse original df with augmentations
        df = self.fuser.fuse(df, augmented_df)

        return df

    def get_aug_idx(self, df, idx):
        return df.loc[idx, self.col_name]

    '''
    retrieves the specified augmented image  
    '''
    def augment_image(self, base_image, aug_idx):
        if aug_idx == 0:
            return base_image
        else:
            return self._calc_augmentation(base_image, aug_idx)

    def _calc_augmentation(self, image, aug):
        raise NotImplementedError()


class CustomAugmentor(BaseAugmentor):
    def __init__(self, fuser, augments=DEFAULT_AUGMENTS):
        super().__init__(augment_cnt=len(augments), col_name='aug', fuser=fuser)

        self.augmentation_dict = augments

    def _calc_augmentation(self, image, aug):
        return self.augmentation_dict[aug](image)

