import os
import tensorflow as tf
model = tf.keras.applications.EfficientNetB0()
model.compile()
tf.keras.models.save_model(
    model, 'broken',
    options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))

model = tf.keras.models.load_model("broken")

from tensorflow.keras import models
import cv2
tf.get_logger().setLevel('ERROR')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

class ObjectDetector:
    def __init__(self, model_path):
        self.model = models.load_model(model_path)

    def detect_objects(self, image_path):
        # Read and preprocess the image
        image = cv2.imread(image_path)
        input_tensor = tf.convert_to_tensor([image], dtype=tf.uint8)  # Convert to tf.uint8
        input_tensor = input_tensor[:, :, :, ::-1]  # Convert BGR to RGB

        # Run the model for object detection
        detections = self.model(input_tensor)

        # Extract relevant information from detections
        boxes = detections['detection_boxes'].numpy()
        scores = detections['detection_scores'].numpy()
        classes = detections['detection_classes'].numpy()

        # Set a confidence threshold
        confidence_threshold = 0.5
        selected_boxes = boxes[scores > confidence_threshold]

        # Return detected objects or bounding boxes
        return selected_boxes
