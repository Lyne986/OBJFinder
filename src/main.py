import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms import functional as F
import cv2

class PeopleHighlightApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load the DeepLabV3+ model
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.eval()

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
        self.image = self.load_image(self.image_path)

        # Initial segmentation and mask
        self.segmentation = self.apply_segmentation(self.image)
        self.update_mask()

    def load_image(self, image_path):
        # Load your image without using cv2.cvtColor or NumPy
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as BGR

        # Convert the image to RGB format without using NumPy
        for y in range(len(image)):
            for x in range(len(image[0])):
                image[y][x][0], image[y][x][2] = image[y][x][2], image[y][x][0]

        return image

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

    def update_mask(self):
        # Update the mask based on the current alpha slider value
        self.alpha = self.alpha_slider.value()
        self.mask = self.create_people_mask(self.segmentation)

        # Color and blur people's bodies with the specified alpha and blur level
        colored_people_image = self.color_and_blur_people(self.image, self.mask, self.alpha, self.blur_level)
        self.display_image(colored_people_image)

    def update_blur_level(self):
        # Update the blur level based on the current blur slider value
        self.blur_level = self.blur_slider.value()

        # Color and blur people's bodies with the specified alpha and blur level
        colored_people_image = self.color_and_blur_people(self.image, self.mask, self.alpha, self.blur_level)
        self.display_image(colored_people_image)

    def color_and_blur_people(self, image, mask, alpha, blur_level):
        # Change the color and alpha of people's bodies without using NumPy
        image_with_colored_people = []

        # Normalize alpha channel to the range [0, 1]
        alpha_normalized = alpha / 255.0

        height, width = len(image), len(image[0])

        for y in range(height):
            row = []
            for x in range(width):
                if mask[y][x]:
                    # Convert the list to a tuple before multiplication
                    color_tuple = (0, 0, 255)
                    updated_pixel = tuple(
                        int((1 - alpha_normalized) * image[y][x][c] +
                            alpha_normalized * color_tuple[c])
                        for c in range(3)
                    )
                    row.append(updated_pixel)
                else:
                    row.append(image[y][x])
            image_with_colored_people.append(row)

        # Apply blur to the detected people's region
        image_with_colored_people = self.apply_blur(image_with_colored_people, mask, blur_level)

        return image_with_colored_people

    def apply_blur(self, image, mask, blur_level):
        # Apply blur to the detected people's region without using NumPy
        if blur_level > 0:
            # Find the bounding box of the people's region
            min_x, min_y, max_x, max_y = self.get_people_bbox(mask)

            # Apply blur to the bounding box region
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if mask[y][x]:
                        # Blur only the people's region
                        image[y][x] = self.apply_pixel_blur(image, x, y, blur_level)

        return image

    def apply_pixel_blur(self, image, x, y, blur_level):
        # Apply blur to a single pixel using a simple averaging filter
        height, width = len(image), len(image[0])
        channels = len(image[0][0])

        total_pixels = 0
        total_color = [0] * channels

        for i in range(max(0, y - blur_level), min(height, y + blur_level + 1)):
            for j in range(max(0, x - blur_level), min(width, x + blur_level + 1)):
                total_pixels += 1
                total_color = [total_color[c] + image[i][j][c] for c in range(channels)]

        averaged_color = [int(total_color[c] / total_pixels) for c in range(channels)]

        return tuple(averaged_color)

    def get_people_bbox(self, mask):
        # Find the bounding box of the people's region
        rows = [any(mask[y]) for y in range(len(mask))]
        cols = [any(mask[y][x] for y in range(len(mask))) for x in range(len(mask[0]))]

        min_x = cols.index(True)
        max_x = len(cols) - 1 - cols[::-1].index(True)
        min_y = rows.index(True)
        max_y = len(rows) - 1 - rows[::-1].index(True)

        return min_x, min_y, max_x, max_y

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PeopleHighlightApp()
    window.show()
    sys.exit(app.exec_())
