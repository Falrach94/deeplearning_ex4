import time
import numpy as np
import torch as t
import torch.cuda


class AutoEncTrainer:
    # --- interface --------------

    def single_epoch_with_eval(self):
        old_time = time.time_ns()

        # train model
        train_loss, train_time = self.train_epoch()
        if self.abort_fit:
            return None

        # validate
        eval_loss, val_time, metrics = self.val_test()

        # calculate combined time
        total_time = (time.time_ns() - old_time) / 10 ** 9

        if self.abort_fit:
            return None

        return {'train': train_loss.item(), 'val': eval_loss.item()}, \
               {'total': total_time, 'train': train_time, 'val': val_time}, \
            metrics

    def train_with_early_stopping(self, max_epoch, patience=10, window=5, early_stop_criterion=None, best_metric_sel=None):
        best_crit_val = -1
        best_epoch = None
        best_model = None
        best_metric = None
        best_loss = None

        best_metric_val = None
        best_metric_model = None

        for i in range(max_epoch):
            loss, time, metrics = self.single_epoch_with_eval()
            if best_metric_sel is not None:
                val, update = best_metric_sel(metrics, best_metric_val)
                if update:
                    best_metric_val = val
                    best_metric_model = self._model.state_dict()

            if early_stop_criterion is None:
                crit = loss['val']
                update = crit < best_crit_val
            else:
                crit, update = early_stop_criterion(loss, metrics, best_crit_val)

            if update or i == 0:
                best_loss = loss['val']
                best_metric = metrics
                best_epoch = i
                best_crit_val = crit
                best_model = self._model.state_dict()

            if self.epoch_callback is not None:
                self.epoch_callback(i, loss, time, metrics,
                                    {'epoch': best_epoch, 'loss': best_loss, 'metric': best_metric})

            if i >= patience:
                if i - best_epoch > window:
                    break

        return best_model, best_metric_model


    # --- setup ------------------

    def __init__(self):
        self._model = None
        self._optim = None
        self._train_dl = None
        self._val_test_dl = None
        self._val_sample_cnt = 0
        self._batch_size = 0

        self.last_metric = None

        self.abort_fit = False

        self.loss_fct = self.calc_loss
        self.val_loss_fct = self.calc_loss

        self.metric_calculator = None
        self.batch_callback = None
        self.epoch_callback = None
        self.epoch = 0

    def set_session(self, model, optim, tr_dl, val_dl, batch_size):
        self.epoch = 0
        self._model = model

        self._optim = optim
        self._train_dl = tr_dl
        self._val_test_dl = val_dl
        self._val_sample_cnt = len(val_dl) * batch_size
        self._batch_size = batch_size

        self._model.cuda()

        self.abort_fit = False

    # --- training primitives ----

    def calc_loss(self, input, pred, label):
        return self._crit(pred, label)

    def train_epoch(self):
        self.epoch += 1
        if hasattr(self._model, 'set_epoch'):
            self._model.set_epoch(self.epoch)

        self._model.train()

        total_loss = 0

        start_time = None
        for i, (x, y) in enumerate(self._train_dl):
            if start_time is None:
                start_time = time.time_ns()

            x = x.cuda()
            y = y.cuda()

            self._optim.zero_grad()
            prediction = self._model(x)
            loss = self.loss_fct(x, prediction, y, self.last_metric)
            loss.backward()
            self._optim.step()
            total_loss += loss

            if self.batch_callback is not None:
                self.batch_callback(i,
                                    len(self._train_dl),
                                    (time.time_ns() - start_time) / 10 ** 9,
                                    True)
            if self.abort_fit:
                return torch.zeros(1), 0

        # training finished
        self.batch_callback(len(self._train_dl),
                            len(self._train_dl),
                            (time.time_ns() - start_time) / 10 ** 9,
                            True)

        av_loss = total_loss / len(self._train_dl)
        return av_loss, (time.time_ns() - start_time) / 10 ** 9

    def val_test(self):

        predictions = torch.empty(self._val_sample_cnt, 2)
        labels = torch.empty(self._val_sample_cnt, 2)

        loss = 0

        self._model.eval()
        with t.no_grad():
            start_time = None

            for i, (x, y) in enumerate(self._val_test_dl):
                if start_time is None:
                    start_time = time.time_ns()

                x = x.cuda()
                y = y.cuda()

                # perform a validation step
                step_prediction = self._model(x)
                loss += self.val_loss_fct(x, step_prediction, y, self.last_metric)

                j = i*self._batch_size
                if type(step_prediction) is tuple:
                    predictions[j:j+y.shape[0]] = step_prediction[1]
                else:
                    predictions[j:j+y.shape[0]] = step_prediction
                labels[j:j+y.shape[0]] = y

                if self.batch_callback is not None:
                    self.batch_callback(i,
                                        len(self._val_test_dl),
                                        (time.time_ns() - start_time) / 10 ** 9,
                                        False)
                if self.abort_fit:
                    return torch.tensor(0), 0, None

        if self.metric_calculator is not None:
            metrics = self.metric_calculator(predictions, labels)
        else:
            metrics = None

        self.last_metric = metrics

        # finished validation loop
        self.batch_callback(len(self._val_test_dl),
                            len(self._val_test_dl),
                            (time.time_ns() - start_time) / 10 ** 9,
                            False)

        av_loss = loss / len(self._val_test_dl)

        return av_loss, (time.time_ns() - start_time) / 10 ** 9, metrics

