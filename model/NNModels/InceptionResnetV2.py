from torch import nn
import torchvision as tv
from torch.nn import init


class InceptionV3(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = tv.models.inception_v3(weights=tv.models.Inception_V3_Weights.DEFAULT)

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 2),
            nn.Sigmoid()
        )
        self.model.AuxLogits.fc = nn.Sequential(
            nn.Linear(768, 2),
            nn.Sigmoid()
        )

        for module in self.model.fc.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)


    def forward(self, x):
        x = self.model(x)

        return x