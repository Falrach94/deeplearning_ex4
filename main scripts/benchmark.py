import torch

from model.NNModels.ResNet50v2 import ResNet50v2
from model.NNModels.ResNet50v2_pre import ResNet50v2_Pretrained

model = ResNet50v2_Pretrained()
model.cuda()

batch_size = 64

data = torch.randn(batch_size, 3, 300, 300, device='cuda')

repetitions = 100
total_time = 0
with torch.no_grad():
    for rep in range(repetitions):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        _ = model(data)

        ender.record()
        torch.cuda.synchronize()
        total_time += starter.elapsed_time(ender)/1000

Throughput = (repetitions*batch_size)/total_time
print('Final Throughput:', Throughput, '1/s')
