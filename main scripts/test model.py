import torch

from model.NNModels.MultipathResnet import MultipathResNet34
from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from model.NNModels.TestResNet import TestResNet34

model = MultipathResNet34(5)
#model = TestResNet34()
model.cuda()

def rand_val(cnt):
    x = torch.rand(cnt, 8, 1, 300, 300).cuda()
    x = x - torch.min(x)
    x = x / torch.max(x)
    x = 2 * (x - 0.5)
    x = x.repeat((1, 1, 3, 1, 1))
    return x
def rand_label(cnt):
    x = torch.rand(cnt, 8, 2).cuda()
    x = x - torch.min(x)
    x = x / torch.max(x)
    return x

optim = torch.optim.Adam(lr=0.00001, params=model.parameters())
loss = torch.nn.MSELoss().cuda()

x_test = rand_val(1)
val = [None] * 5

def eval(path, itarget):
    results = []
    l = []
    model.set_path(path, False)
    model.eval()
    if path == itarget:
        val[path] = None

    for i in range(1):
        pred = model(x_test[i])
        print(pred)
        if val[path] is None:
            results.append(pred)
        else:
            if not torch.equal(pred, val[path][i]):
                print(f'error at net {path}')
                exit()
    if val[path] is None:
        val[path] = results


for i in range(5):
    print('train net', i)
    model.set_path(path=i, train=True)
    model.train()
    cnt = 10
    a = rand_val(cnt)
    b = rand_label(cnt)
    for j in range(cnt):
        optim.zero_grad()
        yp = model(a[j])
      #  if i == 0:
        l = loss(yp, b[j])
        l.backward()
        optim.step()
    print('eval')
    for k in range(1):
        print("rep", k)
        for j in range(5):
            print('eval', j)
            eval(j, i)
print('success')
exit()

for i in range(25):
    x = torch.rand(8, 3, 300, 300)
    x = x.cuda()

    model.set_path(path=0, train=False)
    model.eval()
    y = model(x)

    model.set_path(path=1, train=True)
    model.train()
    for _ in range(1):
        a = torch.rand(32, 3, 300, 300)
        a = x.cuda()
        yp = model(a)
        l = loss(yp, torch.rand_like(yp).cuda())
        l.backward()
        optim.step()

    model.set_path(path=0, train=False)
    model.eval()
    y2 = model(x)
    if not torch.equal(y, y2):
        print(f'failed at it {i+1} of 25')
        exit()

    print(f'{i+1} / 25')

print('success')
