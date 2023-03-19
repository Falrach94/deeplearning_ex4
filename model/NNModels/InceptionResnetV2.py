from torch import nn
import torchvision as tv
from torch.nn import init


class InceptionV3(nn.Module):

    def __init__(self):
        self.model = tv.models.inception_v3(weights=tv.models.Inception_V3_Weights.DEFAULT)

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 4),
            nn.Softmax(dim=1)
        )

        for module in self.model.fc.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)


    def forward(self, x):
        return self.model(x)