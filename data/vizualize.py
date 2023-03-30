import torch
from matplotlib import pyplot as plt

stats = torch.load('../assets/stats.stats')

plt.plot(stats['train'], label='tr')
plt.plot(stats['val'], label='val')

plt.legend()
plt.show()

