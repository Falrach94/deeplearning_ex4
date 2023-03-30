import copy
import time

import numpy as np
import pandas as pd
import torch

from cli_program.config_updater import ConfigUpdater, ConfigUpdateField
from cli_program.settings.behaviour_settings import Modes
from cli_program.settings.config_init import initialize_state, initialize_training_state, initialize_model_state
from cli_program.settings.configurations import default_config

from cli_program.ui import CLInterface
from model.training.genericTrainer import GenericTrainer
from utils.averageApproximator import AverageApproximator
from utils.utils import export, save_eval_stats


class Program:
    def __init__(self):
        self.trainer = None
        self._start_time = None
        self._losses = None
        self._approx = [AverageApproximator(), AverageApproximator()]

        self.config = default_config()
        self.state = initialize_state(self.config)

        self.cli = CLInterface()
        self.cli.prepare_ui(self.state, self.config)


    def _prepare_training(self):
        self.trainer = GenericTrainer()
        self.trainer.set_batch_callback(self._batch_callback)
        self.trainer.set_epoch_callback(self._epoch_callback)
        self.trainer.set_training_loss_calculator(self.state['training']['loss']['tr'].calc)
        self.trainer.set_validation_loss_calculator(self.state['training']['loss']['val'].calc)
        self.trainer.set_metric_calculator(self.config['training']['metric']['calculator'])
        self.trainer.set_metric_selector(self.config['training']['metric']['selector'])
        self.trainer.set_stopping_criterium(None)

    def _batch_callback(self, batch_ix, batch_cnt, time, training):
        if batch_ix == batch_cnt:
            return

        tpb = self._approx[0 if training else 1].add(val=time/(batch_ix + 1))

        approx_rem = tpb * (batch_cnt - batch_ix-1)

        self.cli.batch_update(epoch=self.trainer.epoch,
                              training=training, batch_ix=batch_ix, batch_cnt=batch_cnt,
                              approx_rem=approx_rem, tpb=tpb)

    def _epoch_callback(self, epoch, loss, epoch_time, metrics, best):
        total_time_s = int((time.time_ns() - self._start_time) / 10 ** 9)
        total_time_min = int(total_time_s / 60)
        total_time_s %= 60
        self._losses['train'] += [loss['train']]
        self._losses['val'] += [loss['val']]
        self._losses['metric'] += [metrics]
      #  torch.save(self.state['model'].state_dict(), self.config['path']['ckp'])

        metrics = self._losses['metric'] if metrics is not None else None
        self.cli.epoch_update(epoch, self._losses, epoch_time, metrics, best, (total_time_min, total_time_s))

    def _run_split_training(self):
        model_state, _, _ = self._perform_training()

        torch.save(model_state, self.config['path']['ckp'])
        export(self.state['model'],
               model_state,
               self.config['path']['export'], self.cli.sb)


    def _run_kfold_evaluation(self):

        modifications = self.config['behaviour']['config']['updates']
        updater = ConfigUpdater(self.config, modifications)

        stats = np.zeros((len(updater), 4))

        self.cli.print_eval_table(modifications, stats)
        save_eval_stats('assets/eval.csv', modifications, stats)

        self.cli.print_kfold_table(self.state['data']['folds'],
                                   self.config['behaviour']['config']['k'],
                                   [], [], reset_loc=True)

        for i, config in enumerate(updater):

            mean_loss = []
            mean_f1 = []

            self.cli.reset_progress_bars()
            self.cli.sb.remove_mark('epoch')

            for fold in self.state['data']['folds']:
                self.state['data']['split'] = fold
                self.state['model'] = initialize_model_state(config)
                self.state['training'] = initialize_training_state(self.state, config)

                #preload
               # tr_set = list(fold['tr']['dataset'])
               # val_set = list(fold['val']['dataset'])

                _, loss, f1 = self._perform_training()
                mean_loss += [loss]
                mean_f1 += [f1]
                self.cli.print_kfold_table(self.state['data']['folds'],
                                           self.config['behaviour']['config']['k'],
                                           mean_loss, mean_f1,
                                           reset_loc=False)


            stats[i, 0] = np.mean(mean_loss)
            stats[i, 1] = np.std(mean_loss)
            stats[i, 2] = np.mean(mean_f1)
            stats[i, 3] = np.std(mean_f1)

            self.cli.print_eval_table(modifications, stats)
            save_eval_stats('assets/eval.csv', modifications, stats)

    def run(self):
        switch = {
            Modes.Split: self._run_split_training,
            Modes.KFold: self._run_kfold_evaluation
        }
        switch[self.config['behaviour']['mode']]()

    def _perform_training(self):
        self._start_time = time.time_ns()
        self._prepare_training()
        model = self.state['model'].cuda()
        data = self.state['data']['split']

        self.trainer.set_session(model=model,
                                 optim=self.state['training']['optimizer'],
                                 tr_dl=data['tr']['dl'],
                                 val_dl=data['val']['dl'],
                                 val_cnt=len(data['val']['dataset']),
                                 label_cnt=self.state['data_processor']['labeler'].class_count(True))

        self._losses = {'train': [], 'val': [], 'metric': []}
        best_model_state, best_loss, best_metric = self.trainer.train_with_early_stopping(self.config['training']['config']['max_epoch'],
                                                                     self.config['training']['config']['patience'],
                                                                     self.config['training']['config']['window'])
        return best_model_state, best_loss, best_metric


if __name__ == '__main__':
    print('')
    prog = Program()
    prog.run()
