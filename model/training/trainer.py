import time
import numpy as np
import torch as t
import torch.cuda


class Trainer:

    # --- interface --------------
    def single_epoch_with_eval(self):
        old_time = time.time_ns()

        # train model
        train_loss, train_time = self.train_epoch()
        if self.abort_fit:
            return None

        # validate
        eval_loss, stat_c, stat_i, f1, val_time = self.val_test()

        # calculate combined time
        total_time = (time.time_ns() - old_time) / 10 ** 9

        if self.abort_fit:
            return None

        for i in range(4):
            stat_c[i] = stat_c[i].item()
            stat_i[i] = stat_i[i].item()
        f1 = f1.item()
        eval_loss = eval_loss.item()
        train_loss = train_loss.item()

        self._session.add_epoch(train_loss, eval_loss,
                                total_time,
                                {'state_dict': self._model.state_dict()},
                                f1, stat_c[0], stat_i[0])

        return {'train': train_loss, 'val': eval_loss}, \
            stat_c, stat_i, f1, \
            {'total': total_time, 'train': train_time, 'val': val_time}

    # --- setup ------------------

    def __init__(self):
        self._model = None
        self._crit = None
        self._optim = None
        self._train_dl = None
        self._val_test_dl = None

        self._session = None

        self.abort_fit = False

        self.batch_callback = None

    def set_session(self, session):
        self._session = session
        self._model = session.params.model
        self._crit = session.params.loss

        self._model.cuda()
        self._optim = session.params.optimizer
        self._train_dl = session.params.training_dl
        self._val_test_dl = session.params.eval_dl

        self._crit.cuda()

        self.abort_fit = False

# --- training primitives ----

    def train_epoch(self):
        epoch = self._session.epoch_cnt()
        if hasattr(self._model, 'set_epoch'):
            self._model.set_epoch(epoch)

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
            loss = self._crit(prediction, y)
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
        self.batch_callback(i,
                            len(self._train_dl),
                            (time.time_ns() - start_time) / 10 ** 9,
                            True)

        av_loss = total_loss / len(self._train_dl)
        return av_loss, (time.time_ns() - start_time)/10**9

    def val_test(self):

        batch_size = self._session.batch_size
        sample_cnt = batch_size * len(self._val_test_dl)

        predictions = torch.empty(sample_cnt, 2)
        labels = torch.empty(sample_cnt, 2)
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
                loss += self._crit(step_prediction, y)

                # save the predictions and the labels for each batch
                j = i*batch_size
                predictions[j:j+y.shape[0]] = step_prediction
                labels[j:j+y.shape[0]] = y

                if self.batch_callback is not None:
                    self.batch_callback(i,
                                        len(self._val_test_dl),
                                        (time.time_ns() - start_time) / 10 ** 9,
                                        False)
                if self.abort_fit:
                    return torch.tensor(0), torch.zeros(4), torch.zeros(4), torch.tensor(0), 0

        # finished validation loop
        self.batch_callback(len(self._val_test_dl)+1,
                            len(self._val_test_dl),
                            (time.time_ns() - start_time) / 10 ** 9 ,
                            False)

        av_loss = loss / len(self._val_test_dl)
        stat_c, stat_i, f1 = self.calc_multi_f1(predictions, labels)

        return av_loss, stat_c, stat_i, f1, (time.time_ns() - start_time)/10**9

# --- utils ------------------
    @staticmethod
    def calc_multi_f1(prediction, label):

        stat_c = Trainer.calc_f1(prediction[:, 0], label[:, 0])
        stat_i = Trainer.calc_f1(prediction[:, 1], label[:, 1])
        f1_i = stat_i[0]
        f1_c = stat_c[0]
        f1 = (f1_c + f1_i)/2

        return stat_c, stat_i, f1

    @staticmethod
    def calc_f1(pred, label):
        pred = pred > 0.5
        label = label > 0.5

        tp = np.logical_and(pred, label)
        tn = np.logical_and(np.invert(pred), np.invert(label))
        fp = np.logical_and(pred, np.invert(label))
        fn = np.logical_and(np.invert(pred), label)

        tp = tp.sum()
        tn = tn.sum()
        fp = fp.sum()
        fn = fn.sum()

        if tp + fp == 0:
            precision = np.nan
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = np.nan
        else:
            recall = tp/(tp + fn)

        if np.isnan(recall) or np.isnan(precision) or precision + recall == 0:
            return [torch.tensor(0), tp, tn, fp, fn]

        return [2*precision*recall/(precision+recall), tp, tn, fp, fn]
