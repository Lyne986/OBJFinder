
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, QCheckBox
from PyQt5.QtGui import QImage, QPixmap
from image_modification import ImageModification
from segmentation_model import PeopleSegmentationModel

class ImageWidget(QWidget):
    # Define a signal to notify the parent about deletion
    deleted = pyqtSignal()

    def __init__(self, segmentation_model, parent=None):
        super(ImageWidget, self).__init__(parent)

        # Assign the segmentation model
        self.segmentation_model = segmentation_model

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

        # Connect slider value change to update mask and blur level dynamically
        self.alpha_slider.valueChanged.connect(self.update_mask)

        self.alpha_checkbox = QCheckBox("Show Alpha Slider")
        self.alpha_checkbox.setChecked(True)
        self.alpha_checkbox.stateChanged.connect(self.toggle_alpha_slider)

        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 10)
        self.blur_slider.setValue(self.blur_level)

        # Connect blur slider value change to update blur level dynamically
        self.blur_slider.valueChanged.connect(self.update_blur_level)

        self.blur_checkbox = QCheckBox("Show Blur Slider")
        self.blur_checkbox.setChecked(True)
        self.blur_checkbox.stateChanged.connect(self.toggle_blur_slider)

        # Button to choose an image
        self.choose_image_button = QPushButton("Choose Image")
        self.choose_image_button.clicked.connect(self.choose_image)

        # Button to delete the current image
        self.delete_image_button = QPushButton("Delete Image")
        self.delete_image_button.clicked.connect(self.delete_image)
        self.delete_image_button.setEnabled(False)

        # Layout
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.alpha_checkbox)
        slider_layout.addWidget(QLabel("Alpha:"))
        slider_layout.addWidget(self.alpha_slider)
        
        slider_layout.addWidget(self.blur_checkbox)
        slider_layout.addWidget(QLabel("Blur Level:"))
        slider_layout.addWidget(self.blur_slider)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.choose_image_button)
        button_layout.addWidget(self.delete_image_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(slider_layout)
        main_layout.addLayout(button_layout)

        # Image data
        self.image_path = None
        self.image = None
        self.segmentation = None

    def choose_image(self):
        # Open a file dialog to choose an image
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.jpeg)")
        file_dialog.setWindowTitle("Choose an Image")
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            # Get the selected file path
            selected_file = file_dialog.selectedFiles()[0]

            # Load the selected image
            self.image_path = selected_file
            self.image = self.image_modification.load_image(self.image_path)

            # Initial segmentation and mask
            self.segmentation = self.segmentation_model.apply_segmentation(self.image)

            # Enable sliders and delete button
            self.alpha_slider.setEnabled(True)
            self.blur_slider.setEnabled(True)
            self.delete_image_button.setEnabled(True)

            # Display the selected image
            self.update_mask()

    def delete_image(self):
        # Emit a signal to inform the parent to remove this widget
        self.deleted.emit()

    def update_mask(self):
        # Get the current value of the alpha slider
        current_alpha = self.alpha_slider.value()

        # Check if the alpha slider triggered the update
        if current_alpha != self.alpha:
            # Update the mask based on the current alpha slider value
            self.alpha = current_alpha
            self.mask = self.image_modification.create_people_mask(self.segmentation)

            # Color people's bodies with the specified alpha
            colored_people_image = self.image_modification.color_people(
                self.image, self.mask, self.alpha
            )

            # Save the colored image for future reference
            self.colored_image = colored_people_image

            # Apply blur to the colored image
            self.update_blur_level()

    def update_blur_level(self):
        # Update the blur level based on the current blur slider value
        self.blur_level = self.blur_slider.value()

        # Apply blur to the previously colored image (if it exists)
        if hasattr(self, 'colored_image'):
            colored_blurred_people_image = self.image_modification.apply_blur(
                self.colored_image, self.mask, self.blur_level
            )
            self.display_image(colored_blurred_people_image)

    def toggle_alpha_slider(self, state):
        # Enable/disable alpha slider based on the checkbox state
        self.alpha_slider.setEnabled(state == Qt.Checked)

    def toggle_blur_slider(self, state):
        # Enable/disable blur slider based on the checkbox state
        self.blur_slider.setEnabled(state == Qt.Checked)

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