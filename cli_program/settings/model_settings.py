from cli_program.settings.data_settings import LABEL_COLUMNS
from model.NNModels.ResNet34_pre import ResNet34_Pretrained

MODEL = ResNet34_Pretrained(len(LABEL_COLUMNS))

