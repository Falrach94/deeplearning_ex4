import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.Datasets.autoencoder_dataset import AutoencoderDataset
from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from model.NNModels.VotingNet import VotingNet
from model.reader.small_reader import SmallDataReader
from model.training.autoEncTrainer import AutoEncTrainer
from utils.console_util import print_progress_bar
from utils.loss_utils import calc_BCE_loss
from utils.stat_tools import calc_multi_f1, calc_multi_f1_conf

val_time_per_batch = [1]

def batch_callback(batch_ix, batch_cnt, time, training):
    global val_time_per_batch
    if batch_ix == batch_cnt:
        print()
        return

    tpb = time / (batch_ix + 1)
    val_time_per_batch.append(tpb)
    if len(val_time_per_batch) > 10:
        val_time_per_batch = val_time_per_batch[1:11]
    tpb = np.mean(val_time_per_batch)

    approx_rem = tpb * (batch_cnt - batch_ix - 1)

    print_progress_bar(f'{"training" if training else "validation"}',
                       batch_ix + 1, batch_cnt,
                       f'~{int(approx_rem)} s remaining (~{round(tpb, 2)} s/batch)')


print('loading model')
model = ResNet34_Pretrained()
model.load_state_dict(torch.load('assets/best_model1.ckp'))
#VotingNet.create().cuda()

print('preparing data')
reader = SmallDataReader(prune=False)
tr_data, val_data = reader.get_csv_data(None)
data = pd.concat((tr_data, val_data))

val_set = AutoencoderDataset(data, 'val', 0, True)
val_dl = DataLoader(val_set, 16, False)

print('preparing evaluation')
trainer = AutoEncTrainer(True)
trainer.metric_calculator = calc_multi_f1_conf
trainer.val_loss_fct = calc_BCE_loss
trainer.set_session(model, None, None, val_dl, 16)
trainer.batch_callback = batch_callback

print('evaluating...')
loss, time, metric = trainer.val_test()
print('finished')
print(metric)
