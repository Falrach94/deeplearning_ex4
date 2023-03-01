import sys

from PyQt6.QtWidgets import QApplication

from presenter.prep_presenter import PrepPresenter
from presenter.presenter import Presenter

app = QApplication(sys.argv)
presenter = PrepPresenter()
app.exec()
