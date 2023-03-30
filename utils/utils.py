import pandas as pd
import torch
from torch import nn

from data.utils import combine


def export(model: nn.Module, state, path, sb):
    if sb is not None:
        sb.print_line('starting export')
    else:
        print('starting export')

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
                      opset_version=11,  # the ONNX version to export the model to
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


def save_eval_stats(path, mods: list, vals: list):
    cols = ['#'] + [mod.keys[-1] for mod in mods] + ['loss', 'loss_std', 'f1', 'f1_mean']
    combs = combine([mod.range for mod in mods])
    values = [[i] + c + list(v) for i, (c, v) in enumerate(zip(combs, vals))]

    df = pd.DataFrame(columns=cols, data=values)
    df.to_csv(path, index=False)

