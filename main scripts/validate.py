import pandas as pd

from cli_program.settings.behaviour_settings import DIST_PATH, DEF_PATH
from cli_program.settings.data_settings import VAL_TRANSFORMS
from cli_program.settings.training_settings import BATCH_SIZE, LOSS_CALCULATOR
from data.dataset_generator import create_dataset
from data.image_loader import ImageLoader
from data.label_provider import SimpleLabeler
from model.NNModels.ResNet34_pre import ResNet34Combined
from model.training.genericTrainer import GenericTrainer
from utils.loss_utils import calc_MSE_loss
from utils.stat_tools import calc_f1_m

model = ResNet34Combined(DIST_PATH, DEF_PATH).cuda()


image_loader = ImageLoader(image_path_col='filename')
label_provider = SimpleLabeler('crack', 'inactive')

df = pd.read_csv('assets/val_data.csv')
df = label_provider.label_dataframe(df)
data = create_dataset(df, image_loader, label_provider, None, VAL_TRANSFORMS, BATCH_SIZE, None)

trainer = GenericTrainer()
trainer.set_validation_loss_calculator(calc_MSE_loss)
trainer.set_metric_calculator(calc_f1_m)
trainer.set_session(model, None, None, data['dl'], len(data['dataset']), 2)

print(trainer.val_test())

