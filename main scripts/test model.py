import torch

from model.NNModels.MultipathResnet import MultipathResNet34
from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from model.NNModels.TestResNet import TestResNet34

model = MultipathResNet34(5)
#model = TestResNet34()
model.cuda()

x = torch.rand(32, 3, 300, 300)
x = x.cuda()


optim = torch.optim.Adam(lr=0.001, params=model.parameters())
loss = torch.nn.MSELoss().cuda()



for i in range(100):
    model.set_path(path=0, train=False)
    model.eval()
    y = model(x)

    model.set_path(path=1, train=True)
    model.train()
    yp = model(x)
    l = loss(yp, torch.rand_like(yp).cuda())
    l.backward()

    model.set_path(path=0, train=False)
    model.eval()
    y2 = model(x)
    if not torch.equal(y, y2):
        print(f'failed at it {i} of 100')
        exit()

print('success')
