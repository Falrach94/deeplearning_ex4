import torch
import torch.nn as nn

from model.NNModels.ResNet50_pre import ResNet50_Pretrained
from model.NNModels.ResNet50v2_pre import ResNet50v2_Pretrained


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pred_i = ResNet50v2_Pretrained()
        self.pred_c = ResNet50v2_Pretrained()

        path_i = 'assets/good_models/ResNet50v2_722_889.ckp'
        path_c = 'assets/good_models/ResNet50v2_864_542.ckp'

        cp_i = torch.load(path_i)
        cp_c = torch.load(path_c)

        self.pred_i.load_state_dict(cp_i['state_dict'])
        self.pred_c.load_state_dict(cp_c['state_dict'])

        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x):
        y_c = self.pred_c(x)
        y_c[:, 1] = 0
        y_i = self.pred_i(x)
        y_i[:, 0] = 0
        return y_c+y_i

        #return torch.concat((y_c[:, :1], y_i[:, 1:]), dim=1)
