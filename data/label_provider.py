import torch


class SimpleLabeler:
    def __init__(self, *col_names):
        self.col_names = col_names

    def label_dataframe(self, df):
        label_series = self._get_df_labels(df, self.col_names)
        df['label'] = label_series
        return df

    @staticmethod
    def get_label(df, idx):
        return torch.Tensor(df.loc[idx, 'label'])

    @staticmethod
    def _get_df_labels(data, col_names):
        return list(zip(*[data[col].astype('float') for col in col_names]))

