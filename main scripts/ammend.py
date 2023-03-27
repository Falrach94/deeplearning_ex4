import pandas as pd

from cli_program.settings.data_settings import DATA_PATH
from data.data_filter import OnlyDefectsFilter
from data.dataset_generator import create_dataset
from data.image_loader import ImageLoader
from model.NNModels.ResNet34_pre import ResNet34Sig

model = ResNet34Sig(1, 'assets/best_model_ex.ckp')

data = pd.read_csv(DATA_PATH)
#data = OnlyDefectsFilter.filter(data)


image_provider = ImageLoader()

data = create_dataset(data, image_provider, )