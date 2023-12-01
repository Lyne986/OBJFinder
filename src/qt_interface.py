# qt_interface.py
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from image_modification import ImageModification
from segmentation_model import PeopleSegmentationModel

class QTInterface(QWidget):
    def __init__(self):
        super().__init__()

        # Load the DeepLabV3+ model
        self.segmentation_model = PeopleSegmentationModel()

        # Image modification utility
        self.image_modification = ImageModification()

        # Initialize mask, alpha, and blur level
        self.mask = None
        self.alpha = 128
        self.blur_level = 0

        # UI elements
        self.image_label = QLabel()
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(self.alpha)

        # Connect slider value change to update mask dynamically
        self.alpha_slider.valueChanged.connect(self.update_mask)

        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 10)
        self.blur_slider.setValue(self.blur_level)

        # Connect blur slider value change to update blur level dynamically
        self.blur_slider.valueChanged.connect(self.update_blur_level)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(QLabel("Alpha:"))
        layout.addWidget(self.alpha_slider)
        layout.addWidget(QLabel("Blur Level:"))
        layout.addWidget(self.blur_slider)
        self.setLayout(layout)

        # Set a fixed size for the main window
        self.setFixedSize(800, 600)

        # Load the image
        self.image_path = "./data/sample_images/sample_01.jpeg"
        self.image = self.image_modification.load_image(self.image_path)

        # Initial segmentation and mask
        self.segmentation = self.segmentation_model.apply_segmentation(self.image)
        self.update_mask()

    def update_mask(self):
        # Update the mask based on the current alpha slider value
        self.alpha = self.alpha_slider.value()
        self.mask = self.image_modification.create_people_mask(self.segmentation)

        # Color and blur people's bodies with the specified alpha and blur level
        colored_people_image = self.image_modification.color_and_blur_people(
            self.image, self.mask, self.alpha, self.blur_level
        )
        self.display_image(colored_people_image)

    def update_blur_level(self):
        # Update the blur level based on the current blur slider value
        self.blur_level = self.blur_slider.value()

        # Color and blur people's bodies with the specified alpha and blur level
        colored_people_image = self.image_modification.color_and_blur_people(
            self.image, self.mask, self.alpha, self.blur_level
        )
        self.display_image(colored_people_image)

    def display_image(self, image):
        # Display the image in the QLabel without using NumPy
        height, width = len(image), len(image[0])
        bytes_per_line = 3 * width
        q_image = QImage()

        # Convert the list of lists to a flat list of RGB values
        pixel_values = [val for row in image for pixel in row for val in pixel]

        # Convert the flat list to bytes
        byte_array = bytes(pixel_values)

        q_image = QImage(byte_array, width, height, bytes_per_line, QImage.Format_RGB888)

        # Set aspect ratio policy to keep the image unscaled
        self.image_label.setPixmap(QPixmap(q_image))
        self.image_label.setScaledContents(True)
