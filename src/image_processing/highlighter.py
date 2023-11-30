import cv2
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
class ObjectHighlighter:
    def __init__(self):
        pass

    def highlight_objects(self, image, detected_objects):
        result_image = image.copy()

        # Convert image to grayscale
        gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

        # Define a list of colors for each object
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Green, Red, Blue (add more colors as needed)

        for i, box in enumerate(detected_objects):
            # Convert coordinates to pixel values
            y_min, x_min, y_max, x_max = (box * np.array([result_image.shape[0], result_image.shape[1], result_image.shape[0], result_image.shape[1]])).astype(int)

            # Draw contours on the result image
            color = colors[i % len(colors)]  # Cycle through colors

            # Draw contours around the object
            roi = gray_image[y_min:y_max, x_min:x_max]
            _, thresh = cv2.threshold(roi, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw polygons around the contours
            for contour in contours:
                contour = contour + np.array([x_min, y_min])  # Offset the contour based on the object position
                cv2.polylines(result_image, [contour], isClosed=True, color=color, thickness=2)

        return result_image