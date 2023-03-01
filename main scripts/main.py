import sys

from PyQt6.QtWidgets import QApplication

from presenter.presenter import Presenter

if __name__ == '__main__':
    app = QApplication(sys.argv)
    presenter = Presenter()
    app.exec()


