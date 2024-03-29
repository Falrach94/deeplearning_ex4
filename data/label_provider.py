import numpy as np
import torch
from torch.nn.functional import one_hot


class SimpleLabeler:
    OM_IDX = 'idx'
    OM_ONE_HOT = 'one_hot'
    OM_RAW = 'raw'
    OM_AUTOENCODE = 'auto'
    OM_NAME = 'name'

    def __init__(self, *col_names, output_mode='raw'):
        self.col_names = col_names
        self.output_mode = output_mode

    def set_output_mode(self, mode):
        self.output_mode = mode

    def get_output_mode(self):
        return self.output_mode

    def label_dataframe(self, df):
        label_series = self._get_df_labels(df, self.col_names)
        label_series_id = self._get_df_labels_id(df, self.col_names)
        df['label'] = label_series
        df['label_id'] = label_series_id
        return df

    def class_count(self, raw):
        if raw:
            return len(self.col_names)
        else:
            return 2**len(self.col_names)

    def get_label(self, df, idx, x):
        if self.output_mode == self.OM_NAME:
            return df.loc[idx, 'filename']
        if self.output_mode == self.OM_AUTOENCODE:
            return x[0:1, :, :]

        return self.get_label_from_row(df.iloc[idx])

    def get_label_from_row(self, row):
        if self.output_mode == self.OM_RAW:
            return torch.tensor(row['label'])

        label = row['label_id']
        if self.output_mode == self.OM_IDX:
            return label

        if self.output_mode == self.OM_ONE_HOT:
            return torch.nn.functional.one_hot(torch.tensor(label), self.class_count(False)).float()

        raise NotImplemented(f'output mode "{self.output_mode}" not recognized!')

    @staticmethod
    def _get_df_labels(data, col_names):
        return list(zip(*[data[col].astype('float') for col in col_names]))

    @staticmethod
    def _get_df_labels_id(data, col_names):
        return np.sum([(2**i)*data[col].astype('int') for i, col in enumerate(col_names)], axis=0)


class AuxLabeler(SimpleLabeler):
    def __init__(self, col_names, aux_col, output_mode='raw'):
        super().__init__(*col_names, output_mode=output_mode)
        self.aux_col = aux_col

        self.output_aux = False

    def set_output_aux(self, v):
        self.output_aux = v

    def get_label_from_row(self, row):
        label = super().get_label_from_row(row)

        if self.output_aux:
            aux_idx = int(row[self.aux_col])
            aux_label = one_hot(torch.tensor(aux_idx),
                                4).float()
            return label, aux_label
        return label

class IndexToOneHotLabeler():
    def __init__(self, col, cnt):
        self.col = col
        self.cnt = cnt

    def get_label(self, df, idx, x):
        label_idx = df.loc[idx, self.col]
        label_idx = torch.tensor(label_idx, dtype=torch.long)
        return one_hot(label_idx, self.cnt).float()

    def label_dataframe(self, df):
        return df


class SingleLabeler(SimpleLabeler):

    def __init__(self, *col_names, output_mode='raw'):
        super().__init__(*col_names, output_mode=output_mode)

    def label_dataframe(self, df):
        label_series = self._get_df_labels(df, self.col_names)
        label_series_id = self._get_df_labels_id(df, self.col_names)
        df['label'] = label_series#[:, None]
        df['label_id'] = label_series_id
        return df

    def class_count(self, raw):
        if raw:
            return 1
        else:
            return 2

    @staticmethod
    def _get_df_labels(data, col_names):
        float_labels = [data[col].astype('float') for col in col_names]
        col_sum = np.sum(float_labels, axis=0)

        return (col_sum > 0).astype('float')[:, None]

    @staticmethod
    def _get_df_labels_id(data, col_names):
        return SingleLabeler._get_df_labels(data, col_names).astype('int')


class LabelerTypes:
    SIMPLE = 'simple'
    SINGLE = 'single'
    AUX = 'aux'
    INDEX_TO_ONEHOT = 'indexed'

class LabelerFactory:
    @staticmethod
    def create(type, state, config):
        if type == LabelerTypes.SIMPLE:
            return SimpleLabeler(*config['cols'], output_mode=config['om'])
        elif type == LabelerTypes.SINGLE:
            return SingleLabeler(*config['cols'], output_mode=config['om'])
        elif type == LabelerTypes.AUX:
            return AuxLabeler(config['cols'], config['aux'], output_mode=config['om'])
        elif type == LabelerTypes.INDEX_TO_ONEHOT:
            return IndexToOneHotLabeler(config['col'], config['cnt'])

        raise NotImplemented('labeler type not recognized')

