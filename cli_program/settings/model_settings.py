from data.model_factory import ModelTypes

MODEL_TYPE = ModelTypes.ResNet50_SigAux
LABEL_CNT = 2
MULTI_LAYER = True
MODEL_CONFIG = {
    'out_cnt': LABEL_CNT,
    'multi_layer': MULTI_LAYER
}


