from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, QFrame
from PyQt5.QtGui import QImage, QPixmap, QColor
import cv2
from image_modification import ImageModification
from segmentation_model import PeopleSegmentationModel

class ImageWidget(QWidget):
    deleted = pyqtSignal()

    def __init__(self, segmentation_model, parent=None):
        super(ImageWidget, self).__init__(parent)

        self.segmentation_model = segmentation_model

        self.image_modification = ImageModification()

        self.mask = None
        self.alpha = 128
        self.red = 0
        self.green = 0
        self.blue = 255
        self.blur_level = 0

        self.img_selected = False

        # UI elements
        self.image_label = QLabel()
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(self.alpha)
        self.alpha_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #DDDDDD;
                height: 7px;
                border-radius: 5px;
            }

            QSlider::handle:horizontal {
                background: #000000;
                width: 10px;
                margin: -5px 0;
                border-radius: 1px;
            }
        """)

        self.red_slider = QSlider(Qt.Horizontal)
        self.red_slider.setRange(0, 255)
        self.red_slider.setValue(self.red)
        self.red_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #FFDDDD;
                height: 7px;
                border-radius: 5px;
            }

            QSlider::handle:horizontal {
                background: #FF0000;
                width: 10px;
                margin: -5px 0;
                border-radius: 1px;
            }
        """)

        self.green_slider = QSlider(Qt.Horizontal)
        self.green_slider.setRange(0, 255)
        self.green_slider.setValue(self.green)
        self.green_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #DDFFDD;
                height: 7px;
                border-radius: 5px;
            }

            QSlider::handle:horizontal {
                background: #00FF00;
                width: 10px;
                margin: -5px 0;
                border-radius: 1px;
            }
        """)

        self.blue_slider = QSlider(Qt.Horizontal)
        self.blue_slider.setRange(0, 255)
        self.blue_slider.setValue(self.blue)
        self.blue_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #DDDDFF;
                height: 7px;
                border-radius: 5px;
            }

            QSlider::handle:horizontal {
                background: #0000FF;
                width: 10px;
                margin: -5px 0;
                border-radius: 1px;
            }
        """)

        self.alpha_slider.valueChanged.connect(self.update_mask)
        self.red_slider.valueChanged.connect(self.update_mask)
        self.blue_slider.valueChanged.connect(self.update_mask)
        self.green_slider.valueChanged.connect(self.update_mask)

        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 10)
        self.blur_slider.setValue(self.blur_level)
        self.blur_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #DDDDDD;
                height: 7px;
                border-radius: 5px;
            }

            QSlider::handle:horizontal {
                background: #000000;
                width: 10px;
                margin: -5px 0;
                border-radius: 1px;
            }
        """)

        self.blur_slider.valueChanged.connect(self.update_blur_level)

        self.choose_image_button = QPushButton("Choose Image")
        self.choose_image_button.clicked.connect(self.choose_image)

        self.delete_image_button = QPushButton("Delete Image")
        self.delete_image_button.clicked.connect(self.delete_image)
        self.delete_image_button.setEnabled(False)

        self.save_image_button = QPushButton("Save Modified Image")
        self.save_image_button.clicked.connect(self.save_modified_image)
        self.save_image_button.setEnabled(False)

        slider_layout = QHBoxLayout()

        slider_layout.addWidget(QLabel("Alpha:"))
        slider_layout.addWidget(self.alpha_slider)

        divider_1 = QFrame()
        divider_1.setFrameShape(QFrame.VLine)
        slider_layout.addWidget(divider_1)
        slider_layout.addWidget(QLabel("Blur Level:"))
        slider_layout.addWidget(self.blur_slider)

        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Red: "))
        color_layout.addWidget(self.red_slider)
        color_layout.addWidget(QLabel("Green: "))
        color_layout.addWidget(self.green_slider)
        color_layout.addWidget(QLabel("Blue: "))
        color_layout.addWidget(self.blue_slider)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.choose_image_button)
        button_layout.addWidget(self.delete_image_button)
        button_layout.addWidget(self.save_image_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(slider_layout)
        main_layout.addLayout(color_layout)
        main_layout.addLayout(button_layout)

        self.image_path = None
        self.image = None
        self.segmentation = None

    def choose_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.jpeg)")
        file_dialog.setWindowTitle("Choose an Image")
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]

            self.image_path = selected_file
            self.image = self.image_modification.load_image(self.image_path)
            print(self.image_path)

            self.segmentation = self.segmentation_model.apply_segmentation(self.image)

            self.alpha_slider.setEnabled(True)
            self.red_slider.setEnabled(True)
            self.blue_slider.setEnabled(True)
            self.green_slider.setEnabled(True)
            self.blur_slider.setEnabled(True)
            self.delete_image_button.setEnabled(True)
            self.save_image_button.setEnabled(True)

            self.img_selected = True
            self.update_mask()

    def delete_image(self):
        self.deleted.emit()

    def update_mask(self):
        current_alpha = self.alpha_slider.value()
        current_red = self.red_slider.value()
        current_blue = self.blue_slider.value()
        current_green = self.green_slider.value()

        if current_alpha != self.alpha or current_red != self.red or current_blue != self.blue or current_green != self.green:
            if self.img_selected is False:
                return

            self.alpha = current_alpha
            self.red = current_red
            self.green = current_green
            self.blue = current_blue
            self.mask = self.image_modification.create_people_mask(self.segmentation)
            print(self.alpha)
            print(self.red)
            print(self.blue)
            print(self.green)

            colored_people_image = self.image_modification.color_people(
                self.image, self.mask, self.alpha, self.red, self.green, self.blue
            )

            self.colored_image = colored_people_image

            self.update_blur_level()

    def update_blur_level(self):
        self.blur_level = self.blur_slider.value()

        if hasattr(self, 'colored_image'):
            colored_blurred_people_image = self.image_modification.apply_blur(
                self.colored_image, self.mask, self.blur_level
            )
            self.display_image(colored_blurred_people_image)

    def display_image(self, image):
        height, width = len(image), len(image[0])
        bytes_per_line = 3 * width
        q_image = QImage()

        pixel_values = [val for row in image for pixel in row for val in pixel]

        byte_array = bytes(pixel_values)

        q_image = QImage(byte_array, width, height, bytes_per_line, QImage.Format_RGB888)

        self.image_label.setPixmap(QPixmap(q_image))
        self.image_label.setScaledContents(True)

    def save_modified_image(self):
        if hasattr(self, 'colored_image'):
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("Images (*.png)")
            file_dialog.setWindowTitle("Save Modified Image")
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)

            if file_dialog.exec_():
                save_path = file_dialog.selectedFiles()[0]

                height, width = len(self.colored_image), len(self.colored_image[0])
                bytes_per_line = 3 * width
                q_image = QImage(width, height, QImage.Format_RGB888)

                for y in range(height):
                    for x in range(width):
                        pixel = self.colored_image[y][x]
                        q_color = QColor(pixel[0], pixel[1], pixel[2])
                        q_image.setPixel(x, y, q_color.rgb())

                q_image.save(save_path)