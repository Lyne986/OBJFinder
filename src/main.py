import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms import functional as F
import numpy as np
import cv2

class PeopleHighlightApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load the DeepLabV3+ model
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.eval()

        # UI elements
        self.image_label = QLabel()
        self.segment_button = QPushButton("Highlight People")
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(128)

        # Connect button click to the highlight function
        self.segment_button.clicked.connect(self.highlight_people)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.segment_button)
        layout.addWidget(QLabel("Alpha:"))
        layout.addWidget(self.alpha_slider)
        self.setLayout(layout)

        # Set a fixed size for the main window
        self.setFixedSize(800, 600)

    def load_image(self, image_path):
        # Load your image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    def apply_segmentation(self, image):
        image_tensor = F.to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image_tensor)['out']

        segmentation = output.argmax(1).squeeze().detach().cpu().numpy()
        return segmentation

    def create_people_mask(self, segmentation, people_class_index=15):
        # Create a mask for people's bodies
        mask = segmentation == people_class_index
        return mask

    def color_people_bodies(self, image, mask, alpha):
        # Change the color and alpha of people's bodies
        image_with_colored_people = np.copy(image)

        # Normalize alpha channel to the range [0, 1]
        alpha_normalized = alpha / 255.0

        # Apply alpha blending using NumPy
        image_with_colored_people[mask] = (
            (1 - alpha_normalized) * image_with_colored_people[mask] +
            alpha_normalized * np.array([0, 0, 255])
        ).astype(np.uint8)

        return image_with_colored_people

    def display_image(self, image):
        # Display the image in the QLabel without scaling
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_image)
        
        # Set aspect ratio policy to keep the image unscaled
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def highlight_people(self):
        # Load the image
        image_path = "./data/sample_images/sample_01.jpeg"
        image = self.load_image(image_path)

        # Apply segmentation
        segmentation = self.apply_segmentation(image)

        # Create a mask for people's bodies
        mask = self.create_people_mask(segmentation)

        # Get the alpha value from the slider
        alpha = self.alpha_slider.value()

        # Color people's bodies with the specified alpha
        colored_people_image = self.color_people_bodies(image, mask, alpha)

        # Display the result
        self.display_image(colored_people_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PeopleHighlightApp()
    window.show()
    sys.exit(app.exec_())
