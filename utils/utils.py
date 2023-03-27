import torch
from torch import nn

def export(model: nn.Module, state, path, sb):
    sb.print_line('starting export')

    was_cuda = next(model.parameters()).is_cuda

    if state is not None:
        model.load_state_dict(state)

    model.cpu()
    model.eval()

    x = torch.randn(1, 3, 300, 300, requires_grad=True)
    y = model(x)
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      path + '.zip',  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    if was_cuda:
        model.cuda()

    if sb is not None:
        sb.print_line(f'export finished ("{path}")')
    else:
        print(f'export finished ("{path}")')


def mirror_and_rotate(x, hor, ver, rot):
    if hor:
        x = torch.flipud(x)
    if ver:
        x = torch.fliplr(x)
    if rot != 0:
        x = torch.rot90(x, rot, dims=(1, 2))
    return x

def mirror_horizontal(x):
    x = torch.flipud(x)
    return x


def mirror_vertical(x):
    x = torch.fliplr(x)
    return x

def rotate90deg(x, k):
    x = torch.rot90(x, k, dims=(1, 2))
    return x