from cli_program.settings.data_settings import LABEL_COLUMNS
from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from model.NNModels.ResNet50v2_pre import ResNet50v2_Pretrained

MODEL = ResNet34_Pretrained(len(LABEL_COLUMNS))
#MODEL = ResNet50v2_Pretrained()

