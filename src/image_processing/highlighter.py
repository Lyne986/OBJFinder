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
                cv2.drawContours(result_image, [contour], -1, color, 2)

        return result_image
    
    def apply_segmentation(self, image, segmentation_result):
        # Reshape the segmentation result
        segmentation_result = np.squeeze(segmentation_result)
        segmentation_result = np.moveaxis(segmentation_result, -1, 0)  # Move channels to the first axis

        print("segmentation result: ", segmentation_result)
        # Thresholding
        threshold = 0.1
        binary_mask = (segmentation_result > threshold).astype(np.uint8) * 255

        print("binary mask :", binary_mask)
        # Use only the first channel of the binary mask
        binary_mask_gray = binary_mask[0]

        # Find contours
        contours, _ = cv2.findContours(binary_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print("APPLYING SEG Shapes :", segmentation_result.shape)
        print("APPLYING SEG gray mask shape :", binary_mask_gray.shape)
        print("APPLYING SEG len contour :", len(contours))

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image to draw contours on
        colored_image = image.copy()

        print("APPLYING SEG Shapes : ", binary_mask.shape)
        print("APPLYING SEG gray mask shape : ", binary_mask_gray.shape)
        # Draw contours on the colored image
        print("APPLYING SEG len contour : ", len(contours))
        cv2.drawContours(colored_image, contours, -1, (0, 255, 0), 2)  # Try different thickness values


        return colored_image