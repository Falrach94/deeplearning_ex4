import PyQt6
import pandas as pd
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import QListWidget, QFileDialog

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gui.HyperParamterWidget import HyperParameterWidget
from gui.HyperParamterWidgetBool import HyperParameterWidgetBool
from gui.Slider import Slider
from model.profiles.builder.data_readers import DataReaders
from model.profiles.builder.losses import Losses
from model.profiles.builder.models import Models
from model.profiles.builder.optimizers import Optimizers
from model.profiles.training_configuration import TrainingConfiguration
from model.profiles.training_profile import TrainingProfile
from model.profiles.training_session import Session
from utils.ConfigChangedArgs import ConfigChangedArgs
from utils.ListChangedArgs import ListChangedArgs
from utils.gui_tools import add_vlayout
from utils.stat_tools import calc_profile_f1, calc_data_stats


class CustomCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax_loss = fig.add_subplot()

        super(CustomCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    # --- signals -------------------------
    start_multi_fit = pyqtSignal(int, int)
    train_signal = pyqtSignal()
    close_signal = pyqtSignal()
    config_changed = pyqtSignal(ConfigChangedArgs)
    create_profile = pyqtSignal()
    profile_selection_changed = pyqtSignal(int)
    select_session = pyqtSignal(int)
    export_model = pyqtSignal(int)
    clear_session = pyqtSignal()
    signal_remove_session = pyqtSignal(int)
    signal_clone_model = pyqtSignal(str)
    signal_clone_data = pyqtSignal(str)
    signal_validate = pyqtSignal()
    # --- slots ---------------------------


    @pyqtSlot(bool)
    def on_export_state_changed(self, running):
        self.gb_state.setEnabled(not running)

    @pyqtSlot(list)
    def profiles_updates(self, profiles):
        self.plot_accuracies(profiles)

    @pyqtSlot(int, int)
    def job_time_update(self, sec, epoch):
        min = int(sec/60)
        h = int(min/60)
        min = min % 60
        sec = sec % 60
        self.time_label.setText(f"remaining: ~{h}h {min}min {sec}s (~{epoch} s per epoch)")

    @pyqtSlot(bool)
    def fit_status_changed(self, active):
        if active:
            self.fit_button.setText("Stop")
        else:
            self.fit_button.setText("Fit")
            self.time_label.setText("")

    @pyqtSlot(Session)
    def session_changed(self, args):
        if args.type == ListChangedArgs.ADDED:
            self.session_list.addItem(args.data.get_name())
        if args.type == ListChangedArgs.UPDATED:
            self.update_session(args.data)
            if args.index != -1 and self.session_list.count() >= args.index:
                self.session_list.item(args.index).setText(args.data.get_name())
        if args.type == ListChangedArgs.REMOVED:
            self.session_list.takeItem(args.index)
        if args.type == ListChangedArgs.RESET:
            self.session_list.clear()
            for s in args.data:
                self.session_list.addItem(s.get_name())

    @pyqtSlot(TrainingProfile)
    def profiles_added(self, profile):
        self.profile_list.addItem(profile.name)
        checkbox = QtWidgets.QCheckBox(profile.name)
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(self.acc_cb_state_changed)
        self.acc_layout.addWidget(checkbox)
        self.acc_cbs.append(checkbox)

    @pyqtSlot(TrainingConfiguration)
    def config_update(self, config):
        self.refresh_config_category('opt', config.optimizer, self._opt_param_widgets, self.opt_layout, self.optimizer_cb)
        self.refresh_config_category('loss', config.loss, self._loss_param_widgets, self.loss_layout, self.loss_cb)
        self.refresh_config_category('model', config.model, self._model_param_widgets, self.model_layout, self.model_cb)
        self.refresh_config_category('reader', config.data, self._data_param_widgets, self.data_layout, self.data_cb)

        self.config_name_label.setText(config.get_name())
        self.create_profile_button.setEnabled(config.is_complete())

    @pyqtSlot(int, TrainingProfile)
    def profile_selected(self, i, profile):
        self.profile_list.setCurrentRow(i)

    @pyqtSlot(int, int, int, bool)
    def on_batch_complete(self, i, cnt, remaining, training):
        if training:
            self.label_phase.setText('training')
        else:
            self.label_phase.setText('validation')

        if i < 0:
            self.label_batch.setText('preparing')
            self.label_time.setText('')
        elif i < cnt:
            self.label_batch.setText(f'current batch: {i}/{cnt}')
            self.label_time.setText(f'time remaining: ~{remaining} s')
        else:
            self.label_batch.setText('calculating metrics')
            self.label_time.setText('')

    # --- handler --------------------------

    def button_clone_data_clicked(self):
        dlg = QFileDialog()
       # dlg.setFileMode(QFileDialog.AnyFile)
       # dlg.setFilter("Numpy Data File (*.npy)")

        if dlg.exec_():
            filenames = dlg.selectedFiles()
            if len(filenames) != 0:
                self.signal_clone_data.emit(filenames[0])

    def button_clone_model_clicked(self):
        filename = QFileDialog.getOpenFileName(self, 'Open file', filter="Checkpoint (*.ckp)")
        self.signal_clone_model.emit(filename[0])

    def remove_profile_clicked(self):
        self.signal_remove_session.emit(self.profile_list.currentRow)

    def acc_cb_state_changed(self, checked):
        self.plot_accuracies(self._profiles)

    def session_selection_changed(self, index):
        self.select_session.emit(index)

    def model_selection_changed(self, txt):
        self.config_changed.emit(ConfigChangedArgs('model', txt, None, None))

    def opt_selection_changed(self, txt):
        self.config_changed.emit(ConfigChangedArgs('opt', txt, None, None))

    def data_selection_changed(self, txt):
        self.config_changed.emit(ConfigChangedArgs('reader', txt, None, None))

    def loss_selection_changed(self, txt):
        self.config_changed.emit(ConfigChangedArgs('loss', txt, None, None))

    def hp_changed(self, category, index, value):
        self.config_changed.emit(ConfigChangedArgs(category, None, index, value))

    def closeEvent(self, event):
        self.close_signal.emit()

    def create_profile_clicked(self):
        self.create_profile.emit()

    def change_profile_selection(self):
        self.profile_selection_changed.emit(self.profile_list.currentRow())

    def click_fit(self):
        self.start_multi_fit.emit(self.session_slider.value, self.epoch_slider.value)

    # --- private methods ------------------

    def update_session(self, session):
        self.plot_session(session)
        if session.epoch_cnt() == 0:
            return

        self.label_current.setText(f'cur: (c: {round(session.f1_crack[-1],3)}, i: {round(session.f1_inactive[-1],3)}, m: {round(session.f1[-1],3)})')
        self.label_best_mean.setText(f'best m: ({round(session.best_f1_m[0],3)}, {round(session.best_f1_m[1],3)}, {round(session.best_f1_m[2],3)})')
        self.label_best_crack.setText(f'best c: ({round(session.best_f1_c[0],3)}, {round(session.best_f1_c[1],3)}, {round(session.best_f1_c[2],3)})')
        self.label_best_inactive.setText(f'best i: ({round(session.best_f1_i[0],3)}, {round(session.best_f1_i[1],3)}, {round(session.best_f1_i[2],3)})')

        stats = calc_data_stats(session)
        self.label_data_stat_t_nbr.setText(str(stats[0][0]))
        self.label_data_stat_v_nbr.setText(str(stats[0][1]))
        self.label_data_stat_t_f.setText(str(stats[1][0]))
        self.label_data_stat_v_f.setText(str(stats[1][1]))
        self.label_data_stat_t_c.setText(str(stats[2][0]))
        self.label_data_stat_v_c.setText(str(stats[2][1]))
        self.label_data_stat_t_i.setText(str(stats[3][0]))
        self.label_data_stat_v_i.setText(str(stats[3][1]))
        self.label_data_stat_t_b.setText(str(stats[4][0]))
        self.label_data_stat_v_b.setText(str(stats[4][1]))

    def refresh_config_category(self, cat, descriptor, widgets, layout, combo_box):

        # remove hyperparameters if descriptor is None, doesn't contain hp or type selection was changed
        if descriptor is None or descriptor.hyperparams is None \
        or (combo_box is not None and combo_box.currentText() != descriptor.name)\
        or len(widgets) != len(descriptor.hyperparams):
            for w in widgets:
                w.value_changed.disconnect(self.hp_changed)
                layout.removeWidget(w)
            widgets.clear()

        # reset combo_box and return if descriptor is None
        if descriptor is None:
            combo_box.setCurrentIndex(-1)
            return

        if combo_box is not None:
            combo_box.setCurrentText(descriptor.name)

        if descriptor.hyperparams is None:
            return

        if len(descriptor.hyperparams) != len(widgets):
            for i, param in enumerate(descriptor.hyperparams):
                if param.type == 'bool':
                    pw = HyperParameterWidgetBool(cat, i, param)
                else:
                    pw = HyperParameterWidget(cat, i, param)
                pw.value_changed.connect(self.hp_changed)
                layout.addWidget(pw)
                widgets.append(pw)
        else:
            for i, param in enumerate(descriptor.hyperparams):
                widgets[i].set_value(param.get_value())

    def plot_session(self, session):
        self.canvas.ax_loss.clear()

        self.last_session = session

        tr_loss = session.training_loss
        val_loss = session.eval_loss

        if len(tr_loss) > 15:
            tr_loss = tr_loss[5:]
            val_loss = val_loss[5:]
        #if len(tr_loss) > 10:
        #    tr_loss = tr_loss[10:]
        #    val_loss = val_loss[10:]

        training_loss = pd.Series(tr_loss).rolling(self.rolling_average).mean()
        eval_loss = pd.Series(val_loss).rolling(self.rolling_average).mean()
        self.canvas.ax_loss.plot(training_loss, label='training loss')
        self.canvas.ax_loss.plot(eval_loss, label='test loss')

        self.canvas.ax_loss.legend()
        self.canvas.draw()

    def plot_accuracies(self, profiles):
        if profiles is None:
            return
        self._profiles = profiles

        self.acc_ax.clear()

        # f1 mean
        once = False
        for i, p in enumerate(profiles):
            if not self.acc_cbs[i].isChecked():
                continue
            once = True
            f1, f1_c, f1_i = calc_profile_f1(p)

            ts = pd.Series(f1)
            data = ts.rolling(self.rolling_average).mean()
            self.acc_ax.plot(data, label=f'{p.name} (mean)')

            if not self.plot_only_mean:
                ts = pd.Series(f1_c)
                data = ts.rolling(self.rolling_average).mean()
                self.acc_ax.plot(data, label=f'{p.name} (crack)')

                ts = pd.Series(f1_i)
                data = ts.rolling(self.rolling_average).mean()
                self.acc_ax.plot(data, label=f'{p.name} (inactive)')

        if once:
            self.acc_ax.legend()
        self.acc_canvas.draw()

    def smooth_slider_value_changed(self, value):
        self.rolling_average = int(value)
        self.plot_accuracies(self._profiles)
        if self.last_session is not None:
            self.plot_session((self.last_session))

    def epoch_slider_value_changed(self, value):
        self.acc_cnt = int(value)
        self.plot_accuracies(self._profiles)

    def button_export_current(self):
        self.export_model.emit(0)

    def button_export_best_m(self):
        self.export_model.emit(3)

    def button_export_best_c(self):
        self.export_model.emit(1)

    def button_export_best_i(self):
        self.export_model.emit(2)

    def button_clear_session_clicked(self):
        self.clear_session.emit()

    def click_validate(self):
        self.signal_validate.emit()

    # --- construction ---------------------

    def load_profile_builder(self):
        self.loss_cb.addItems(Losses.losses)
        self.optimizer_cb.addItems(Optimizers.optimizers)
        self.model_cb.addItems(Models.models)
        self.data_cb.addItems(DataReaders.reader)

    def __init__(self):
        super(MainWindow, self).__init__()

        self.plot_only_mean = False

        self.rolling_average = 1
        self.acc_cnt = 50

        self._profiles = None

        self._opt_param_widgets = []
        self._model_param_widgets = []
        self._data_param_widgets = []
        self._loss_param_widgets = []

        self.acc_figure = Figure()
        self.acc_ax = self.acc_figure.add_subplot()

        self.last_session = None

        # new configuration widgets
        self.model_layout = None
        self.loss_layout = None
        self.opt_layout = None
        self.data_layout = None
        self.config_name_label = None
        self.create_profile_button = None
        self.model_cb = None
        self.loss_cb = None
        self.optimizer_cb = None

        # profile widgets
        self.profile_list = None

        self.label_data_stat_t_nbr = None
        self.label_data_stat_t_f = None
        self.label_data_stat_t_c = None
        self.label_data_stat_t_i = None
        self.label_data_stat_t_b = None
        self.label_data_stat_v_nbr = None
        self.label_data_stat_v_f = None
        self.label_data_stat_v_c = None
        self.label_data_stat_v_i = None
        self.label_data_stat_v_b = None

        # session widgets
        self.session_list = None
        self.session_slider = None
        self.epoch_slider = None
        self.fit_button = None
        self.time_label = None

        self.label_current = None
        self.label_best_mean = None
        self.label_best_crack = None
        self.label_best_inactive = None
        self.gb_state = None
        self.label_batch = None
        self.label_phase = None
        self.label_time = None

        # monitoring widgets
        self.acc_canvas = None
        self.canvas = None
        self.profile_acc_check_gb = None
        self.acc_layout = None
        self.acc_cbs = []


        self.button_clone_model = None
        self.button_clone_data = None

        self.init_widgets()


        # --- horizontal main layout -----

        # ---------------------------------

        self.load_profile_builder()

    def init_config_widgets(self, layout):

        # config_layout

        # --- declarations
        gb_model = QtWidgets.QGroupBox("Model")
        self.model_layout = QtWidgets.QVBoxLayout()
        gb_loss = QtWidgets.QGroupBox("Loss")
        self.loss_layout = QtWidgets.QVBoxLayout()
        gb_opt = QtWidgets.QGroupBox("Optimizer")
        self.opt_layout = QtWidgets.QVBoxLayout()
        gb_data = QtWidgets.QGroupBox("Data")
        self.data_layout = QtWidgets.QVBoxLayout()
        placeholder = QtWidgets.QWidget()
        self.config_name_label = QtWidgets.QLabel()
        self.create_profile_button = QtWidgets.QPushButton("Create Profile")
        self.model_cb = QtWidgets.QComboBox()  # model selection
        self.loss_cb = QtWidgets.QComboBox()  # loss selection
        self.optimizer_cb = QtWidgets.QComboBox()  # optimizer selection
        self.data_cb = QtWidgets.QComboBox()  # model selection

        # --- layout
        layout.addWidget(gb_model)
        self.model_layout.addWidget(self.model_cb)
        layout.addWidget(gb_loss)
        self.loss_layout.addWidget(self.loss_cb)
        layout.addWidget(gb_opt)
        self.opt_layout.addWidget(self.optimizer_cb)
        layout.addWidget(gb_data)
        self.data_layout.addWidget(self.data_cb)
        layout.addWidget(placeholder)
        layout.addWidget(self.config_name_label)
        layout.addWidget(self.create_profile_button)

        # --- initialization
        gb_model.setLayout(self.model_layout)

        gb_loss.setLayout(self.loss_layout)

        gb_opt.setLayout(self.opt_layout)

        gb_data.setLayout(self.data_layout)

        placeholder.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Expanding)

        self.create_profile_button.setEnabled(False)
        self.create_profile_button.clicked.connect(self.create_profile_clicked)

        self.model_cb.currentTextChanged.connect(self.model_selection_changed)

        self.loss_cb.currentTextChanged.connect(self.loss_selection_changed)

        self.optimizer_cb.currentTextChanged.connect(self.opt_selection_changed)
        self.data_cb.currentTextChanged.connect(self.data_selection_changed)

    def init_profile_widgets(self, layout):
        # profile layout

        # --- declarations
        self.profile_list = QListWidget()
        button_remove = QtWidgets.QPushButton('Remove')
        label = QtWidgets.QLabel("Training Profiles")
#        self.acc_canvas = FigureCanvasQTAgg(self.acc_figure)

        gb_data = QtWidgets.QGroupBox('Session Data')
        data_layout = QtWidgets.QGridLayout(gb_data)
        label_r0 = QtWidgets.QLabel('t')
        label_r1 = QtWidgets.QLabel('v')
        label_c0 = QtWidgets.QLabel('#')
        label_c1 = QtWidgets.QLabel('#f')
        label_c2 = QtWidgets.QLabel('#c')
        label_c3 = QtWidgets.QLabel('#i')
        label_c4 = QtWidgets.QLabel('#b')

        self.label_data_stat_t_nbr = QtWidgets.QLabel('0')
        self.label_data_stat_t_f = QtWidgets.QLabel('0')
        self.label_data_stat_t_c = QtWidgets.QLabel('0')
        self.label_data_stat_t_i = QtWidgets.QLabel('0')
        self.label_data_stat_t_b = QtWidgets.QLabel('0')
        self.label_data_stat_v_nbr = QtWidgets.QLabel('0')
        self.label_data_stat_v_f = QtWidgets.QLabel('0')
        self.label_data_stat_v_c = QtWidgets.QLabel('0')
        self.label_data_stat_v_i = QtWidgets.QLabel('0')
        self.label_data_stat_v_b = QtWidgets.QLabel('0')


        # --- layout
        layout.addWidget(label)
        layout.addWidget(self.profile_list)

        layout.addWidget(button_remove)

        layout.addWidget(gb_data)

        data_layout.addWidget(label_r0, 0, 1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(label_r1, 0, 2, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        data_layout.addWidget(label_c0, 1, 0, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(label_c1, 2, 0, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(label_c2, 3, 0, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(label_c3, 4, 0, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(label_c4, 5, 0, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        data_layout.addWidget(self.label_data_stat_t_nbr, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(self.label_data_stat_t_f, 2, 1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(self.label_data_stat_t_c, 3, 1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(self.label_data_stat_t_i, 4, 1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(self.label_data_stat_t_b, 5, 1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        data_layout.addWidget(self.label_data_stat_v_nbr, 1, 2, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(self.label_data_stat_v_f, 2, 2, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(self.label_data_stat_v_c, 3, 2, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(self.label_data_stat_v_i, 4, 2, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(self.label_data_stat_v_b, 5, 2, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        # --- initialization
        self.profile_list.setFixedWidth(300)
        self.profile_list.itemSelectionChanged.connect(self.change_profile_selection)
        button_remove.clicked.connect(self.remove_profile_clicked)

    def init_session_widgets(self, layout, panel):
        # --- declarations
        self.session_list = QListWidget()
        label = QtWidgets.QLabel("Training Sessions")
        gb_training = QtWidgets.QGroupBox('Training')
        self.session_slider = Slider('session #', 1, 100, 1)
        self.epoch_slider = Slider('epoch #', 1, 1000, 10)
        self.fit_button = QtWidgets.QPushButton('Fit')
        self.time_label = QtWidgets.QLabel("")
        clear_button = QtWidgets.QPushButton('Clear')

        self.gb_state = QtWidgets.QGroupBox('Status')
        status_grid = QtWidgets.QGridLayout(self.gb_state)
        self.label_current       = QtWidgets.QLabel('cur: (0.000, 0.000, 0.000)')
        self.label_best_mean     = QtWidgets.QLabel('best m: (0.000, 0.000, 0.000)')
        self.label_best_crack    = QtWidgets.QLabel('best c: (0.000, 0.000, 0.000)')
        self.label_best_inactive = QtWidgets.QLabel('best i: (0.000, 0.000, 0.000)')
        button_export_current = QtWidgets.QPushButton('Export')
        button_export_best_m = QtWidgets.QPushButton('Export')
        button_export_best_c = QtWidgets.QPushButton('Export')
        button_export_best_i = QtWidgets.QPushButton('Export')
        self.label_batch = QtWidgets.QLabel()
        self.label_phase = QtWidgets.QLabel()
        self.label_time = QtWidgets.QLabel()
        val_button = QtWidgets.QPushButton('Validate')

        self.button_clone_model = QtWidgets.QPushButton('Set Checkpoint')
        self.button_clone_data = QtWidgets.QPushButton('Set Data')

        # --- layout
        layout.addWidget(label)
        layout.addWidget(self.session_list)

        layout.addWidget(gb_training)
        training_layout, _ = add_vlayout(layout)
        training_layout.addWidget(self.button_clone_model)
        training_layout.addWidget(self.button_clone_data)
        training_layout.addWidget(self.session_slider)
        training_layout.addWidget(self.epoch_slider)
        training_layout.addWidget(clear_button)
        training_layout.addWidget(val_button)
        training_layout.addWidget(self.fit_button)
        training_layout.addWidget(self.time_label)

        layout.addWidget(self.gb_state)
        status_grid.addWidget(self.label_current, 0, 0)
        status_grid.addWidget(self.label_best_mean, 1, 0)
        status_grid.addWidget(self.label_best_crack, 2, 0)
        status_grid.addWidget(self.label_best_inactive, 3, 0)
        status_grid.addWidget(button_export_current, 0, 1)
        status_grid.addWidget(button_export_best_m, 1, 1)
        status_grid.addWidget(button_export_best_c, 2, 1)
        status_grid.addWidget(button_export_best_i, 3, 1)

        layout.addWidget(self.label_phase)
        layout.addWidget(self.label_batch)
        layout.addWidget(self.label_time)

        # --- initialization
        panel.setFixedWidth(250)

        self.session_list.currentRowChanged.connect(self.session_selection_changed)

        gb_training.setLayout(training_layout)

        self.fit_button.clicked.connect(self.click_fit)
        button_export_current.clicked.connect(self.button_export_current)
        button_export_best_c.clicked.connect(self.button_export_best_c)
        button_export_best_i.clicked.connect(self.button_export_best_i)
        button_export_best_m.clicked.connect(self.button_export_best_m)
        clear_button.clicked.connect(self.button_clear_session_clicked)
        self.button_clone_model.clicked.connect(self.button_clone_model_clicked)
        self.button_clone_data.clicked.connect(self.button_clone_data_clicked)
        val_button.clicked.connect(self.click_validate)
        panel.setFixedWidth(350)

    def init_monitoring_widgets(self, layout):

        # declarations
        tab = QtWidgets.QTabWidget()

        self.canvas = CustomCanvas()

        acc_panel = QtWidgets.QWidget()
        self.acc_layout = QtWidgets.QVBoxLayout()
        self.acc_canvas = FigureCanvasQTAgg(self.acc_figure)
        epoch_slider = Slider("max epoch", 10, 500, self.acc_cnt)
        smooth_slider = Slider("running average cnt", 1, 20, self.rolling_average)
        self.profile_acc_check_gb = QtWidgets.QGroupBox('Visible')

        # layout
        layout.addWidget(tab)
        tab.addTab(self.canvas, "Session Plot")
        tab.addTab(acc_panel, "Accuracies")
        self.acc_layout.addWidget(self.acc_canvas)
        self.acc_layout.addWidget(epoch_slider)


        layout.addWidget(smooth_slider)

        # initializations
        tab.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                          QtWidgets.QSizePolicy.Policy.Ignored)

        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                          QtWidgets.QSizePolicy.Policy.Expanding)
        acc_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                          QtWidgets.QSizePolicy.Policy.Expanding)
        acc_panel.setLayout(self.acc_layout)
        epoch_slider.value_changed.connect(self.epoch_slider_value_changed)
        smooth_slider.value_changed.connect(self.smooth_slider_value_changed)

        smooth_slider.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                    QtWidgets.QSizePolicy.Policy.Fixed)

    def init_widgets(self):
        main = QtWidgets.QWidget()

        self.setCentralWidget(main)
        main_layout = QtWidgets.QHBoxLayout()
        main.setLayout(main_layout)

        config_layout, config_panel = add_vlayout(main_layout)
        profile_layout, profile_panel = add_vlayout(main_layout)
        session_layout, session_panel = add_vlayout(main_layout)
        monitoring_layout, monitoring_panel = add_vlayout(main_layout)

        self.init_config_widgets(config_layout)
        self.init_profile_widgets(profile_layout)
        self.init_session_widgets(session_layout, session_panel)
        self.init_monitoring_widgets(monitoring_layout)

        config_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                    QtWidgets.QSizePolicy.Policy.Expanding)
        profile_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                    QtWidgets.QSizePolicy.Policy.Expanding)
        session_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                    QtWidgets.QSizePolicy.Policy.Expanding)
        monitoring_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                    QtWidgets.QSizePolicy.Policy.Expanding)

