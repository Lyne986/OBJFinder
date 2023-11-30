import cv2
from object_detection.detector import ObjectDetector
from image_processing.highlighter import ObjectHighlighter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # Set paths and parameters
    model_path = './data/model/centernet_hg104_512x512_coco17_tpu-8/saved_model'
    image_path = './data/sample_images/sample_01.jpeg'

    # Initialize Object Detector
    detector = ObjectDetector(model_path)

    # Detect objects in the image
    detected_objects = detector.detect_objects(image_path)
    # detected_objects = [[0.31110087, 0.44024128, 0.7921666, 0.5337339 ], [0.4871961, 0.5335798, 0.78029835, 0.5961493 ],[0.54409474, 0.40709233, 0.70301193, 0.4872657 ]];

    # Load the original image
    original_image = cv2.imread(image_path)

    # Check if the images are loaded and objects are detected successfully
    if original_image is None:
        print(f"Error: Unable to load the original image from path: {image_path}")
        return

    if detected_objects is None:
        print("Error: Object detection failed.")
        return

    # # Filter detected objects to only include those with a 'class_id' key and class ID 1
    detected_people = [obj for obj in detected_objects if 'class_id' in obj and obj['class_id'] == 1]

    print("Detected objects", detected_objects)

    # Initialize Object Highlighter
    highlighter = ObjectHighlighter()

    # Highlight detected people in the original image
    result_image = highlighter.highlight_objects(original_image, detected_objects)

    # Check if the result image is created successfully
    if result_image is None:
        print("Error: Object highlighting failed.")
        return

    # Print the detected people and their bounding boxes
    print("Detected People:", detected_people)

    # Resize the images if they have different dimensions
    if original_image.shape[0] != result_image.shape[0] or original_image.shape[1] != result_image.shape[1]:
        result_image = cv2.resize(result_image, (original_image.shape[1], original_image.shape[0]))

    # Ensure both images have the same number of channels
    if original_image.shape[2] != result_image.shape[2]:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Display the original and result images side by side
    combined_image = cv2.hconcat([original_image, result_image])

    # Display or save the combined image
    cv2.imshow("Original and Result Images", combined_image)

    # Add a key check to close the window when any key is pressed
    cv2.waitKey(0)

    # Release the OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()