import PyQt6
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import QWidget
from PyQt6 import QtWidgets


class HyperParameterWidget(QWidget):

    value_changed = pyqtSignal(str, int, float)

    def __init__(self, cat, i, param):
        super().__init__()
        self.min = param.min
        self.max = param.max
        self.val = None

        self._category = cat
        self._index = i

        self.slider = None
        self.label = None
        self.textbox = None

        self.use_slider = None

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)

        label = QtWidgets.QLabel(param.name)
        layout.addWidget(label)

        if self.min is not None and self.max is not None:
            self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(self.min*100, self.max*100)
            layout.addWidget(self.slider)
            self.label = QtWidgets.QLabel()
            self.slider.valueChanged.connect(self.slider_value_changed)
            layout.addWidget(self.label)
            self.use_slider = True
        else:
            self.textbox = QtWidgets.QLineEdit()
            self.layout().addWidget(self.textbox)
            self.use_slider = False
            self.textbox.editingFinished.connect(self.confirm)

        self.set_value(param.get_value())

    def confirm(self):
        if self.use_slider:
            val = self.slider.value() / 100
        else:
            val = float(self.textbox.text())
        if (self.min is not None and self.min > val) or (self.max is not None and self.max < val):
            tmp = self.val
            self.val = None
            self.set_value(tmp)
        else:
            self.val = val
            self.value_changed.emit(self._category, self._index, self.val)

    def slider_value_changed(self, value):
        self.label.setText(str(value/100))
        self.confirm()

    def set_value(self, value):

        if self.val == value:
            return

        self.val = value
        if self.use_slider:
            self.slider.setValue(value*100)
            self.label.setText(str(value))
        else:
            self.textbox.setText(f"{value}")

    def get_value(self):
        return self.val
