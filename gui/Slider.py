from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget


class Slider(QWidget):
    value_changed = pyqtSignal(int)

    def __init__(self, txt, min, max, default, fl=False):
        super().__init__()

        self.value = None

        layout = QtWidgets.QHBoxLayout()

        self.use_float = fl

        self.text_label = QtWidgets.QLabel(txt)
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.label = QtWidgets.QLabel()

        self.slider.valueChanged.connect(self.slider_value_changed)

        layout.addWidget(self.text_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.set_range(min, max)
        self.set_value(default)

    def set_range(self, min, max):
        if self.use_float:
            self.slider.setRange(min*100, max*100)
        else:
            self.slider.setRange(min, max)

    def get_value(self):
        return self.value

    def set_value(self, val):

        if self.value == val:
            return False
        self.value = val
        self.label.setText(str(self.value))

        if self.use_float:
            val = int(100*val)
        self.slider.setValue(val)
        return True

    def slider_value_changed(self, value):
        if self.use_float:
            value /= 100

        if self.set_value(value):
            self.value_changed.emit(self.value)
