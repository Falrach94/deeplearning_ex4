import numpy as np
import torch.nn

import torchvision as tv

class VGG19_Pretrained(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = tv.models.vgg.vgg19(
            weights=tv.models.VGG19_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier._modules['6'] = torch.nn.Linear(4096, 2)

        self.sigmoid = torch.nn.Sigmoid()

        self.transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, d):
        x = torch.tensor(np.empty((d.shape[0], 3, 224, 224)), dtype=torch.float)
        if d.is_cuda:
            x.cuda()

        for i in range(x.shape[0]):
            tmp = self.transform(d[i])
            if d.is_cuda:
                tmp.cuda()
            x[i, ...] = tmp

        x = self.model(x)
        return self.sigmoid(x)
