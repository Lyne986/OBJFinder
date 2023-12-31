from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, QCheckBox
from PyQt5.QtGui import QImage, QPixmap
from image_modification import ImageModification
from segmentation_model import PeopleSegmentationModel
from Widget.image_widget import ImageWidget
from PyQt5.QtGui import QFont, QFontDatabase

#Made by 디오고 파리아 말틴스 / Diogo Faria Martins #5023192
class QTInterface(QWidget):
    def __init__(self):
        super(QTInterface, self).__init__()

        self.segmentation_model = PeopleSegmentationModel()

        font_id = QFontDatabase.addApplicationFont("./src/Fonts/Roboto-Regular.ttf")
        print(font_id)
        if font_id != -1:
            print("font uploaded")
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            font = QFont(font_family, 12)
            self.setFont(font)

        main_layout = QVBoxLayout(self)
        self.scroll_area = QWidget()

        self.scroll_layout = QVBoxLayout(self.scroll_area)

        main_layout.addWidget(self.scroll_area)

        self.add_image_button = QPushButton("Add Image")
        self.add_image_button.clicked.connect(self.add_image_widget)
        main_layout.addWidget(self.add_image_button)

        self.setFixedSize(1200, 800)

        self.add_image_widget()

    def add_image_widget(self):
        image_widget = ImageWidget(self.segmentation_model)
        image_widget.deleted.connect(lambda: self.remove_image_widget(image_widget))
        self.scroll_layout.addWidget(image_widget)

    def remove_image_widget(self, widget):
        self.scroll_layout.removeWidget(widget)
        widget.setParent(None)
        widget.deleteLater()
