import torch

from model.NNModels.AutoEncoderClassifier import ResNet34AutoEnc
from model.NNModels.MultipathResnet import MultipathResNet34
from model.NNModels.ResNet34_4to2 import ResNet34_4to2
from model.NNModels.ResNet34_pre import ResNet34_Pretrained

best_classifier_path = 'assets/base_model1.ckp'
output_path = 'assets/export'

print('loading state dict')
#state_dict = torch.load(best_classifier_path)

print('creating model')
model = ResNet34_4to2()
#model.load_state_dict(state_dict)
model.eval()
#model.set_path(4, False)
print('starting export')
x = torch.randn(1, 3, 300, 300, requires_grad=True)
y = model(x)
torch.onnx.export(model,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  output_path + '.zip',  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                'output': {0: 'batch_size'}})

print('export finished')

