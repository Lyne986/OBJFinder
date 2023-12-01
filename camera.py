import cv2
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms import functional as F
import numpy as np

# Load a pre-trained DeepLabV3+ model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Allow the camera to warm up
cv2.waitKey(1000)

# Set new width and height for resized frames
new_width, new_height = 640, 480
skip_frames = 2
frame_count = 0

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Skip frames if necessary
    if frame_count % skip_frames == 0:
        # Resize the frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Convert the OpenCV BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PyTorch tensor
        frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0)

        # Perform segmentation
        with torch.no_grad():
            output = model(frame_tensor)['out']

        # Get the segmentation mask
        segmentation = output.argmax(1).squeeze().detach().cpu().numpy()

        # Define the class index for people in the segmentation model
        people_class_index = 15  # Adjust this based on the class index for people in your model

        # Create a mask for people
        people_mask = segmentation == people_class_index

        # Blend blue color with the original frame only where people are detected
        blue_color = np.array([255, 0, 0], dtype=np.uint8)
        frame[people_mask] = frame[people_mask] * 0.7 + blue_color * 0.3

        # Display the result frame
        cv2.imshow('Real-Time Segmentation', frame)

    # Increment frame count
    frame_count += 1

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
