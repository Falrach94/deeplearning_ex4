from PyQt6 import QtWidgets


def add_vlayout(parent_layout):

    panel = QtWidgets.QWidget()
    parent_layout.addWidget(panel)
    layout = QtWidgets.QVBoxLayout()
    panel.setLayout(layout)
    return layout, panel


def add_hlayout(parent_layout):

    panel = QtWidgets.QWidget()
    parent_layout.addWidget(panel)
    layout = QtWidgets.QHBoxLayout()
    panel.setLayout(layout)
    return layout, panel

