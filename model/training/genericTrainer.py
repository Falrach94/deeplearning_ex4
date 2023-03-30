import copy
import time
import torch as t
import torch.cuda

from cli_program.settings.behaviour_settings import BEST_MODEL_PATH, EXPORT_PATH
from cli_program.ui import SINGLETON_SB
from utils.utils import export


class GenericTrainer:
    # --- interface --------------

    def set_batch_callback(self, callback):
        self.batch_callback = callback

    def set_epoch_callback(self, callback):
        self.epoch_callback = callback

    def set_aux_loss_calculator(self, fct):
        self.aux_loss = fct

    def set_training_loss_calculator(self, fct):
        self.loss_fct = fct

    def set_validation_loss_calculator(self, fct):
        self.val_loss_fct = fct

    def set_metric_calculator(self, fct):
        self.metric_calculator = fct

    def set_stopping_criterium(self, fct):
        self.early_stop_criterion = fct
    def set_metric_selector(self, fct):
        self.best_metric_selector = fct

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
    def repeat_eval(self, last_result):
        old_time = time.time_ns()

        # validate
        eval_loss, val_time, metrics = self.val_test()

        # calculate combined time
        total_time = (time.time_ns() - old_time) / 10 ** 9

        if self.abort_fit:
            return None

        last_result[0]['val'] = eval_loss.item()
        last_result[1]['total'] += total_time
        last_result[1]['val'] += val_time

        return (last_result[0], last_result[1], metrics)

    def train_with_early_stopping(self, max_epoch, patience=10, window=5):
        best_crit_val = -1
        best_epoch = None
        best_model = None
        best_metric = None
        best_loss = None

        best_metric_val = None

        for i in range(max_epoch):
            loss, time, metrics = self.single_epoch_with_eval()
            if self.best_metric_selector is not None:
                crit, update = self.best_metric_selector(metrics, best_metric_val)
                if update:
                    best_metric_val = crit
            elif self.early_stop_criterion is None:
                crit = loss['val']
                update = crit < best_crit_val
            else:
                crit, update = self.early_stop_criterion(loss, metrics, best_crit_val)

            if update or i == 0:
                best_loss = loss['val']
                best_metric = metrics
                best_epoch = i
                best_crit_val = crit
                best_model = copy.deepcopy(self._model.state_dict())


            if self.epoch_callback is not None:
                self.epoch_callback(i, loss, time, metrics,
                                    {'epoch': best_epoch, 'loss': best_loss, 'metric': best_metric})


            if i >= patience:
                if i - best_epoch > window:
                    break

        return best_model, best_loss, best_crit_val


    # --- setup ------------------

    def __init__(self, cuda=True):
        self._model = None
        self._optim = None
        self._train_dl = None
        self._val_test_dl = None
        self._val_sample_cnt = 0

        self.use_cuda = cuda

        self.last_metric = None

        self.abort_fit = False

        self.aux_loss = None
        self.loss_fct = None
        self.val_loss_fct = None

        self.early_stop_criterion = None
        self.best_metric_selector = None
        self.metric_calculator = None
        self.batch_callback = None
        self.epoch_callback = None
        self.epoch = 0

    def use_aux(self):
        return self.aux_loss is not None

    def set_session(self, model, optim,
                    tr_dl, val_dl,
                    val_cnt,
                    label_cnt):
        self.epoch = 0
        self._model = model
        self._optim = optim
        self._train_dl = tr_dl
        self._val_test_dl = val_dl
        self._val_sample_cnt = val_cnt
        self._label_cnt = label_cnt

        self.last_metric = None

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

            if self.use_aux():
                y_aux = y[1].cuda()
                y = y[0]

            if len(y.shape) == 1:
                y = y[:, None]

            x = x.cuda()
            y = y.cuda()

            self._optim.zero_grad()
            prediction = self._model(x)
          #  if len(prediction) == 2:
          #      loss = self.loss_fct(x, prediction[0], y, self.last_metric)
          #      loss += self.loss_fct(x, prediction[1], y, self.last_metric)
          #      loss.backward()
          #  else:
            loss = self.loss_fct(x, prediction, y, self.last_metric)

            if self.use_aux():
                loss += self.aux_loss(x,
                                      self._model.aux_prediction,
                                      y_aux,
                                      self.last_metric)

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

        predictions = torch.empty(self._val_sample_cnt, self._label_cnt)
        labels = torch.empty(self._val_sample_cnt, self._label_cnt)

        loss = 0

        self._model.eval()
        with t.no_grad():
            start_time = None

            for i, (x, y) in enumerate(self._val_test_dl):
                if start_time is None:
                    start_time = time.time_ns()

                if len(y.shape) == 1:
                    y = y[:, None]

                if self.use_cuda:
                    x = x.cuda()
                    y = y.cuda()

                # perform a validation step
                step_prediction = self._model(x)

                loss += self.val_loss_fct(x, step_prediction, y, self.last_metric)

                if self.metric_calculator is not None:
                    j = i*step_prediction.shape[0]
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

        if self.batch_callback is not None:
            # finished validation loop
            self.batch_callback(len(self._val_test_dl),
                                len(self._val_test_dl),
                                (time.time_ns() - start_time) / 10 ** 9,
                                False)

        av_loss = loss / len(self._val_test_dl)

        return av_loss, (time.time_ns() - start_time) / 10 ** 9, metrics

