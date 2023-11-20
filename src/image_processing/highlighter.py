import cv2

class ObjectHighlighter:
    def __init__(self):
        pass

    def highlight_objects(self, image, objects):
        if image is None or len(objects) == 0:
            return image

        highlighted_image = image.copy()

        for obj in objects:
            # Check if the 'bbox' key exists
            if 'bbox' in obj:
                bbox = obj['bbox']

                # Check if bbox is a list or tuple with four elements
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x, y, w, h = bbox
                else:
                    print(f"Warning: Skipping object with invalid 'bbox' structure: {bbox}")
                    continue

                # Draw a rectangle around the detected object
                cv2.rectangle(highlighted_image, (int(x * image.shape[1]), int(y * image.shape[0])),
                              (int((x + w) * image.shape[1]), int((y + h) * image.shape[0])), (0, 255, 0), 2)

                # Add a text label
                label = obj.get('class_name', 'Object')
                cv2.putText(highlighted_image, label, (int(x * image.shape[1]), int(y * image.shape[0]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Apply a custom modification to the aspect of the detected object
                # For example, you can draw a border around the object
                border_thickness = 5
                cv2.rectangle(highlighted_image,
                              (int((x - border_thickness) * image.shape[1]), int((y - border_thickness) * image.shape[0])),
                              (int((x + w + border_thickness) * image.shape[1]), int((y + h + border_thickness) * image.shape[0])),
                              (0, 255, 0), border_thickness)
            else:
                print("Warning: Skipping object with missing 'bbox' key:", obj)

        return highlighted_image
