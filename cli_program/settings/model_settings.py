from cli_program.settings.data_settings import LABEL_COLUMNS
from model.NNModels.AutoEncoder import ResNetAutoEncoder
from model.NNModels.InceptionResnetV2 import InceptionV3
from model.NNModels.ResNet34_pre import ResNet34Sig
from model.NNModels.ResNet50v2_pre import ResNet50v2_Pretrained

MODEL = ResNet34Sig(2)
#MODEL = ResNetAutoEncoder()
#MODEL = InceptionV3()
#MODEL = ResNet50v2_Pretrained()

