# main.py
import sys
from PyQt5.QtWidgets import QApplication
from qt_interface import QTInterface

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QTInterface()
    window.show()
    sys.exit(app.exec_())
