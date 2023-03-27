import pandas as pd
import torch

from cli_program.settings.behaviour_settings import DIST_PATH, DEF_PATH, BEST_MODEL_PATH
from cli_program.settings.data_settings import VAL_TRANSFORMS
from cli_program.settings.training_settings import BATCH_SIZE, LOSS_CALCULATOR
from data.data_filter import OnlyDefectsFilter
from data.dataset_generator import create_dataset
from data.image_loader import ImageLoader
from data.label_provider import SimpleLabeler, SingleLabeler
from model.NNModels.InceptionResnetV2 import InceptionV3
from model.NNModels.ResNet34_pre import ResNet34Combined, ResNet34Sig
from model.training.genericTrainer import GenericTrainer
from utils.loss_utils import calc_MSE_loss
from utils.stat_tools import calc_f1_m

df = pd.read_csv('assets/val_data.csv')
image_loader = ImageLoader(image_path_col='filename')


#model_def = ResNet34Sig(2, DEF_PATH).cuda()
def_label_provider = SimpleLabeler('crack', 'inactive')
df_def = df.copy()
df_def = def_label_provider.label_dataframe(df_def)
df_def = OnlyDefectsFilter.filter(df_def)
data_def = create_dataset(df_def, image_loader, def_label_provider, None, VAL_TRANSFORMS, BATCH_SIZE, None)


#model_dist = ResNet34Sig(1, DIST_PATH, multi_layer=False).cuda()
dist_label_provider = SingleLabeler('crack', 'inactive')
df_dist = df.copy()
df_dist = dist_label_provider.label_dataframe(df_dist)
data_dist = create_dataset(df_dist, image_loader, dist_label_provider, None, VAL_TRANSFORMS, BATCH_SIZE, None)

model = InceptionV3()
model.load_state_dict(torch.load(BEST_MODEL_PATH))
#model = ResNet34Combined(DIST_PATH, DEF_PATH).cuda()
label_provider = SimpleLabeler('crack', 'inactive')
df = label_provider.label_dataframe(df)
data = create_dataset(df, image_loader, label_provider, None, VAL_TRANSFORMS, BATCH_SIZE, None)


trainer = GenericTrainer()
trainer.set_validation_loss_calculator(calc_MSE_loss)
trainer.set_metric_calculator(calc_f1_m)

def validate(model, dl, val_len, label_cnt, name):
    trainer.set_session(model, None, None, dl, val_len, label_cnt)
    loss, time, stats = trainer.val_test()
    print(name)
    print(stats)

#validate(model_dist, data_dist['dl'], len(data_dist['dataset']), 1, 'dist')
#validate(model_def, data_def['dl'], len(data_def['dataset']), 2, 'def')
validate(model, data['dl'], len(data['dataset']), 2, 'comb')
