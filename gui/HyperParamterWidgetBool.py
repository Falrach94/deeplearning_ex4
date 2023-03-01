import PyQt6
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import QWidget
from PyQt6 import QtWidgets


class HyperParameterWidgetBool(QWidget):

    value_changed = pyqtSignal(str, int, float)

    def __init__(self, cat, i, param):
        super().__init__()
        self.min = param.min
        self.max = param.max
        self.val = None

        self._category = cat
        self._index = i

        layout = QtWidgets.QHBoxLayout(self)
        self.checkbox = QtWidgets.QCheckBox(param.name)

        layout.addWidget(self.checkbox)

        self.checkbox.stateChanged.connect(self.state_changed)
        self.set_value(param.get_value())

    def state_changed(self, state):
        self.value_changed.emit(self._category, self._index, state)

    def set_value(self, value):
        if self.val == value:
            return
        self.val = value
        self.checkbox.setChecked(value)

    def get_value(self):
        return self.val
