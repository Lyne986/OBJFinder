import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

class ObjectDetector:
    def __init__(self, detection_model_path):
        self.object_detection_model = tf.keras.models.load_model(detection_model_path)
        self.segmentation_model = VGG16(weights='imagenet', include_top=False)

    def preprocess_image(self, image):
        # Add preprocessing steps as needed (resize, normalize, etc.)
        # Example: resized_image = cv2.resize(image, (desired_width, desired_height))
        return image

    def detect_objects(self, image_path):
        # Read and preprocess the image
        image = cv2.imread(image_path)
        preprocessed_image = self.preprocess_image(image)

        # Run the model for object detection
        detections = self.object_detection_model(np.expand_dims(preprocessed_image, axis=0))

        # Extract relevant information from detections
        boxes = detections['detection_boxes'].numpy()
        scores = detections['detection_scores'].numpy()

        # Set a confidence threshold
        confidence_threshold = 0.5
        selected_boxes = boxes[scores > confidence_threshold]

        return selected_boxes

    def perform_segmentation(self, image, box):
        # Extract coordinates
        y_min, x_min, y_max, x_max = box

        # Extract the region of interest
        roi = image[int(y_min * image.shape[0]):int(y_max * image.shape[0]),
                    int(x_min * image.shape[1]):int(x_max * image.shape[1])]

        # Resize the ROI to match VGG16 input size
        roi_resized = cv2.resize(roi, (224, 224))

        # Ensure the input has 3 channels (RGB)
        roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

        # Preprocess the input
        roi_preprocessed = preprocess_input(roi_resized)

        # Perform segmentation
        segmentation_result = self.segmentation_model.predict(np.expand_dims(roi_preprocessed, axis=0))

        return segmentation_result