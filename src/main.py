# main.py
import sys
from PyQt5.QtWidgets import QApplication
from qt_interface import QTInterface

#Made by 디오고 파리아 말틴스 / Diogo Faria Martins #5023192
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QTInterface()
    window.show()
    sys.exit(app.exec_())
