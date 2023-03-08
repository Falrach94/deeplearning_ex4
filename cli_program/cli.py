import time

from cli_program.settings.behaviour_settings import BEST_MODEL_PATH, EXPORT_PATH
from cli_program.settings.data_settings import *
from cli_program.settings.model_settings import *
from cli_program.settings.training_settings import *
from cli_program.ui import CLInterface
from data.augment_generator import CustomAugmentor
from data.dataset_generator import create_single_split_datasets
from data.data_reader import CSVReader
from data.image_loader import AugmentedImageLoader
from data.label_provider import SimpleLabeler
from model.training.genericTrainer import GenericTrainer
from utils.averageApproximator import AverageApproximator
from utils.utils import export


class Program:
    def __init__(self):
        self.data = None
        self.trainer = None
        self.cli = CLInterface()

        self._start_time = None

        self._approx = [AverageApproximator(), AverageApproximator()]

        self._prepare_ui()
        self._prepare_data()
        self._prepare_training()

    def _prepare_data(self):
        augmentor = CustomAugmentor(AUGMENTATIONS)
        image_provider = AugmentedImageLoader(image_path_col='filename',
                                              augmentor=augmentor)
        label_provider = SimpleLabeler(*LABEL_COLUMNS)
        self.data = CSVReader(path=DATA_PATH, seperator=CSV_SEPERATOR).get()
        self.data = label_provider.label_dataframe(self.data)

        self.data = create_single_split_datasets(
            data=self.data,
            split=HOLDOUT_SPLIT,
            image_provider=image_provider,
            label_provider=label_provider,
            augmentor=augmentor,
            tr_transform=TR_TRANSFORMS,
            val_transform=VAL_TRANSFORMS,
            batch_size=BATCH_SIZE
        )

    def _prepare_ui(self):
        self.cli.prepare_ui()

    def _prepare_training(self):
        self.trainer = GenericTrainer()
        self.trainer.set_batch_callback(self._batch_callback)
        self.trainer.set_epoch_callback(self._epoch_callback)
        self.trainer.set_training_loss_calculator(TRAINING_LOSS)
        self.trainer.set_validation_loss_calculator(VALIDATION_LOSS)
        self.trainer.set_metric_calculator(METRIC_CALC)
        self.trainer.set_metric_selector(BEST_METRIC_SELECTOR)
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
        total_time_s = int((time.time_ns() - self.start_time) / 10 ** 9)
        total_time_min = int(total_time_s / 60)
        total_time_s %= 60
        self.cli.epoch_update(epoch, loss, epoch_time, metrics, best, (total_time_min, total_time_s))

    def perform_training(self):
        self._start_time = time.time_ns()
        model = MODEL.cuda()
        self.trainer.set_session(model=model,
                                 optim=OPTIMIZER_FACTORY.create(model.parameters()),
                                 tr_dl=self.data['tr']['dl'],
                                 val_dl=self.data['val']['dl'],
                                 batch_size=BATCH_SIZE)
        best_model_state = self.trainer.train_with_early_stopping(MAX_EPOCH, PATIENCE, WINDOW)
        model.load_state_dict(best_model_state)

        torch.save(best_model_state, BEST_MODEL_PATH)
        export(model, best_model_state, EXPORT_PATH, self.sb)


prog = Program()
prog.perform_training()