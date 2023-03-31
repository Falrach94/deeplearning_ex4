import torch
from matplotlib import pyplot as plt

stats = torch.load('../assets/stats.stats')

#ax = plt.gca()

ar = [v['mean'] for v in stats['metric']]
arc = [v['stats'][0]['f1'] for v in stats['metric']]
ari = [v['stats'][1]['f1'] for v in stats['metric']]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(stats['train'], label='tr')
ax1.plot(stats['val'], label='val')

plt.legend()
ax2.plot(ar, label='f1')
ax2.plot(arc, label='f1_c')
ax2.plot(ari, label='f1_i')


plt.legend()
plt.show()

