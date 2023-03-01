import json
import os
import threading

import numpy as np
import pandas as pd
import torch
from PyQt6.QtCore import pyqtSlot, QObject, pyqtSignal
from skimage.io import imread

from gui.MainWindow import MainWindow, ConfigChangedArgs
from gui.PreProcessingWindow import PreProcessingWindow
from model.NNModels.Fusion import FusionNet
from model.preprocessing.preprocessor import Preprocessor
from model.profiles.builder.data_readers import DataReaders
from model.profiles.builder.losses import Losses
from model.profiles.builder.models import Models
from model.profiles.builder.optimizers import Optimizers
from model.profiles.training_configuration import TrainingConfiguration
from model.profiles.training_profile import TrainingProfile
from model.profiles.training_session import Session
from model.reader.data_reader import DataReader
from model.training.trainer import Trainer
from utils.ListChangedArgs import ListChangedArgs

class Presenter (QObject):

    #  --- signals --------------------
    profile_added = pyqtSignal(TrainingProfile)
    update_signal = pyqtSignal(object, object)
    selected_profile_changed = pyqtSignal(int, object)
    configuration_changed = pyqtSignal(TrainingConfiguration)
    session_changed = pyqtSignal(ListChangedArgs)
    fit_status_changed = pyqtSignal(bool)
    expected_time_update = pyqtSignal(int, int)
    profiles_updated = pyqtSignal(list)
    export_state_change = pyqtSignal(bool)
    batch_complete_signal = pyqtSignal(int, int, int, bool)

    #  --- slots ----------------------
    @pyqtSlot(str)
    def on_window_clone_checkpoint(self, path):
        if self.selected_session is None:
            return
        data = torch.load(path, 'cuda')
        self.selected_session.reset(data)

    @pyqtSlot(str)
    def on_window_clone_data(self, path):
        if self.selected_session is None:
            return

        if path[-6:] == 'tr.npy':
            tr_path = path
            val_path = path[:-6] + 'val.npy'
        else:
            tr_path = path[:-7] + 'tr.npy'
            val_path = path

        self.selected_session.clone_data(tr_path, val_path)

    @pyqtSlot()
    def on_window_clear_sessions(self):
        if self.fit_thread is not None:
            return
        if self.selected_profile is None:
            return

        self.selected_session = None
        self.selected_profile.sessions = []
        self.session_changed.emit(ListChangedArgs(ListChangedArgs.RESET, -1, []))

    @pyqtSlot(int)
    def on_window_session_selection_changed(self, index):
        if self.selected_profile is None:
            return

        self.select_session(index)

    @pyqtSlot()
    def on_window_train(self):
        #self.fit_thread = threading.Thread(target=self.fit)
        #self.fit_thread.start()
        pass

    @pyqtSlot()
    def on_window_close(self):
        self.abort = True

    @pyqtSlot(ConfigChangedArgs)
    def on_window_config_changed(self, args):
        if args.type == 'loss' and args.selection is not None:
            self.new_configuration.loss \
                = Losses.get_descriptor(args.selection) if len(args.selection) != 0 else None
        elif args.type == 'opt' and args.selection is not None:
            self.new_configuration.optimizer \
                = Optimizers.get_descriptor(args.selection) if len(args.selection) != 0 else None
        elif args.type == 'model' and args.selection is not None:
            self.new_configuration.model \
                = Models.get_descriptor(args.selection) if len(args.selection) != 0 else None
        elif args.type == 'reader' and args.selection is not None:
            self.new_configuration.data \
                = DataReaders.get_descriptor(args.selection) if len(args.selection) != 0 else None
        elif args.type == 'reader':
            self.new_configuration.data.hyperparams[args.par_id].set_value(args.par_val)
        elif args.type == 'opt':
            self.new_configuration.optimizer.hyperparams[args.par_id].set_value(args.par_val)
        elif args.type == 'model':
            self.new_configuration.model.hyperparams[args.par_id].set_value(args.par_val)
        elif args.type == 'loss':
            self.new_configuration.loss.hyperparams[args.par_id].set_value(args.par_val)
        else:
            raise NotImplemented()

        self.configuration_changed.emit(self.new_configuration)

    @pyqtSlot()
    def on_window_create_profile(self):
        for i in range(len(self.profiles)):
            if self.profiles[i].configuration == self.new_configuration:
                self.select_profile(i)
                return
        self.add_profile(TrainingProfile(self.new_configuration.clone()))

    @pyqtSlot(int)
    def on_window_profile_selection_changed(self, index):
        self.select_profile(index)

    @pyqtSlot(int, int)
    def on_window_start_multi_fit(self, session_cnt, epoch_cnt):
        if self.fit_thread is not None:
            self.abort = True
            self._trainer.abort_fit = True
            return

        if self.selected_profile is None:
            print("No profile selected!")
            return
        self.batch_complete_signal.emit(-1, 0, 0, True)
        self.create_sessions(session_cnt)
        self.fit_thread = threading.Thread(target=self.multi_fit, args=[session_cnt, epoch_cnt])
        self.fit_thread.start()

    @pyqtSlot(int)
    def on_window_export(self, type):

        print('export started')
        self.export_state_change.emit(True)
        t = threading.Thread(target=self.export_checkpoint, args=[type])
        t.start()

    @pyqtSlot()
    def on_window_validate(self):
        t = threading.Thread(target=self.validate)
        t.start()
    #  --- construction ---------------

    def load_profiles(self):
        pass

    def connect_window_signals(self):
        self._window.close_signal.connect(self.on_window_close)
        self._window.train_signal.connect(self.on_window_train)
        self._window.config_changed.connect(self.on_window_config_changed)
        self._window.create_profile.connect(self.on_window_create_profile)
        self._window.profile_selection_changed.connect(self.on_window_profile_selection_changed)
        self._window.start_multi_fit.connect(self.on_window_start_multi_fit)
        self._window.select_session.connect(self.on_window_session_selection_changed)
        self._window.export_model.connect(self.on_window_export)
        self._window.clear_session.connect(self.on_window_clear_sessions)
        self._window.signal_clone_model.connect(self.on_window_clone_checkpoint)
        self._window.signal_clone_data.connect(self.on_window_clone_data)
        self._window.signal_validate.connect(self.on_window_validate)


    def connect_window_slots(self):
        self.configuration_changed.connect(self._window.config_update)
        self.profile_added.connect(self._window.profiles_added)
        self.session_changed.connect(self._window.session_changed)
        self.fit_status_changed.connect(self._window.fit_status_changed)
        self.expected_time_update.connect(self._window.job_time_update)
        self.profiles_updated.connect(self._window.profiles_updates)
        self.export_state_change.connect(self._window.on_export_state_changed)
        self.batch_complete_signal.connect(self._window.on_batch_complete)

    def __init__(self):
        super().__init__()

        self._window = MainWindow()
        self.connect_window_slots()
        self.connect_window_signals()
        self._window.show()

        self.fit_thread = None
        self._trainer = Trainer()
        self._trainer.batch_callback = self.batch_complete

        self.abort = False

        # profiles
        self.new_configuration = TrainingConfiguration()
        self.profiles = []
        self.load_profiles()
        self.selected_profile = None

        self.remaining_epochs = None

        self.selected_session = None

    # --- interface --------------------

    def set_trainer(self, trainer):
        self._trainer = trainer

    #def show(self):
    #    self.on_window_train()

    #def update(self, train_losses, eval_losses):
    #    if self._trainer is None:
    #        pass

    #    self.update_signal.emit(train_losses, eval_losses)

    # --- private methods --------------

    def select_session(self, index):
        if index == -1:
            self.selected_session = None
            return

        session = self.selected_profile.sessions[index]
        self.selected_session = session
        self.selected_session.load_or_create_data(self.selected_profile.configuration)

        self.session_changed.emit(ListChangedArgs(ListChangedArgs.UPDATED,
                                                  index,
                                                  session))

    def update_expected_job_time(self):
        epoch_time = self.selected_profile.get_average_epoch_time()
        self.expected_time_update.emit(int(self.remaining_epochs*epoch_time), int(epoch_time))

    def load_profile(self, path):
        with open(path, 'r') as file:
            self.add_profile(TrainingProfile.from_json(json.load(file)))

    def load_profiles(self):
        for root, _, files in os.walk('assets/profiles'):
            for name in files:
                if name[-4:] == 'json':
                    path = os.path.join(root, name)
                    self.load_profile(path)

    def batch_complete(self, i, cnt, time, training):
        if i % 10 == 0 or i >= cnt:
            remaining = (cnt-(i+1))*(time / (i+1))
            self.batch_complete_signal.emit(i+1, cnt, int(remaining), training)


    @staticmethod
    def save_profile(profile):
        with open(f'assets/profiles/{profile.name}.json', 'w') as file:
            file.write(profile.to_json())

    def save_profiles(self):
        for profile in self.profiles:
            self.save_profile(profile)

    def multi_fit(self, cnt, epochs):

        self.abort = False

        self.fit_status_changed.emit(True)

        self.remaining_epochs = cnt*epochs
        self.update_expected_job_time()

        for i in range(0, cnt):
            print(f"start fitting session {i+1}/{cnt}")
            self.fit_session(i, epochs)
            if self.abort:
                break

        print(f"fit complete")

        self.fit_thread = None
        self.fit_status_changed.emit(False)

    def validate(self):
        print('validation started')
#        if self.selected_session is None:
#            self.create_sessions(1)
#            self.selected_session = self.selected_profile.sessions[0]
#        session = self.selected_session
#        session.initialize(self.selected_profile.configuration)
        #self._trainer.set_session(session)
        #self._trainer.validate()
        print('validation finished')
        print('exporting model...')

        path = 'assets/exports/fusion'
        #cp_data = cp_data.data

        #model = Models.instantiate(self.selected_profile.configuration.model).cpu()
        #model.load_state_dict(session.params.model)
        #model.eval()

        model = FusionNet()
        model.cpu()

        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = model(x)

        torch.onnx.export(model,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              path+'.zip',                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})


        print('export finished')


    def fit_session(self, i, epochs):
        session = self.selected_profile.sessions[i]
        session.initialize(self.selected_profile.configuration)
        self._trainer.set_session(session)

        self.remaining_epochs -= session.epoch_cnt()

        while session.epoch_cnt() < epochs and not self.abort:
            res = self._trainer.single_epoch_with_eval()
            if res is None:
                continue

            loss, stat_c, stat_i, f1, time = res

            print(f"{session.epoch_cnt()+1}/{epochs} | time  | total: {round(time['total'],1)} s; training: {round(time['train'], 1)} s; validation: {round(time['val'],1)} s;")
            print(f"{session.epoch_cnt()+1}/{epochs} | loss  | training: {round(loss['train'],4)}; validation: {round(loss['val'], 4)}")
            print(f"{session.epoch_cnt()+1}/{epochs} | F1    | crack: {round(stat_c[0], 2)}; inactive: {round(stat_i[0], 2)}; mean {round(f1, 2)}")
            print(f"{session.epoch_cnt()+1}/{epochs} | stats | crack    | tp: {stat_c[1]}; tn: {stat_c[2]}; fp: {stat_c[3]}; fn: {stat_c[4]}")
            print(f"{session.epoch_cnt()+1}/{epochs} | stats | inactive | tp: {stat_i[1]}; tn: {stat_i[2]}; fp: {stat_i[3]}; fn: {stat_i[4]}")

            self.session_changed.emit(ListChangedArgs(ListChangedArgs.UPDATED, index=i, data=session))
            self.save_profile(self.selected_profile)
            self.remaining_epochs -= 1
            self.update_expected_job_time()
            self.profiles_updated.emit(self.profiles)

    def add_profile(self, profile):
        self.profiles.append(profile)
        self.save_profile(profile)
        self.select_profile(len(self.profiles)-1)
        self.profile_added.emit(profile)
        self.profiles_updated.emit(self.profiles)

    def select_profile(self, i):
        if 0 <= i < len(self.profiles):
            self.selected_profile = self.profiles[i]
            self.new_configuration = self.selected_profile.configuration.clone()
        else:
            self.selected_profile = None
            self.new_configuration = TrainingConfiguration()

        self.selected_profile_changed.emit(i, self.selected_profile)
        self.configuration_changed.emit(self.new_configuration)

        self.session_changed.emit(ListChangedArgs(ListChangedArgs.RESET,
                                                  data=[] if self.selected_profile is None
                                                  else self.selected_profile.sessions))

    def create_sessions(self, cnt):
        while len(self.selected_profile.sessions) < cnt:
            session_name = self.selected_profile.name+'#'+str(len(self.selected_profile.sessions))
            session = Session(session_name)
            self.selected_profile.sessions.append(session)
            self.session_changed.emit(ListChangedArgs(ListChangedArgs.ADDED, data=session))

    def export_checkpoint(self, type):

        if self.selected_profile is None\
        or self.selected_session is None:
            print('no profile or session selected')
            self.export_state_change.emit(False)
            return

        rel_path = 'assets/exports/'
        if type == 0:
            cp_data = self.selected_session.checkpoint
        elif type == 1:
            cp_data = self.selected_session.cp_best_f1_c
        elif type == 2:
            cp_data = self.selected_session.cp_best_f1_i
        elif type == 3:
            cp_data = self.selected_session.cp_best_f1_m
        else:
            cp_data = None

        if cp_data is None:
            self.export_state_change.emit(False)
            return

        if not cp_data.is_loaded() and (not cp_data.is_valid() or not cp_data.load()):
            print('Failed to load checkpoint data! (corrupted, erased or empty)')
            self.export_state_change.emit(False)
            return

        path = rel_path + cp_data.name
        cp_data = cp_data.data

        model = Models.instantiate(self.selected_profile.configuration.model).cpu()
        model.load_state_dict(cp_data['state_dict'])
        model.eval()

        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = model(x)

        torch.onnx.export(model,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              path+'.zip',                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})

        torch.save(cp_data, path + '.ckp')

        print('export finished')
        self.export_state_change.emit(False)