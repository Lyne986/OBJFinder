import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load a pre-trained DeepLabV3+ model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# Load your image using OpenCV
image_path = "./data/sample_images/sample_01.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Read image with alpha channel

# Ensure the image has an alpha channel
if image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

image_rgb = image[:, :, :3]  # Extract RGB channels

image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)['out']

segmentation = output.argmax(1).squeeze().detach().cpu().numpy()

# Create a mask for people's bodies
people_class_index = 15
mask = segmentation == people_class_index

# Change the color and alpha of people's bodies
color = np.array([255, 0, 0, 128], dtype=np.uint8)  # Red color with alpha
alpha = 128.0  # Set transparency level (0 to 255) as a float

# Extract the alpha channel
alpha_channel = image[:, :, 3]  # Extract the alpha channel

# Normalize alpha channel to the range [0, 1]
alpha_normalized = alpha_channel / 255.0

# Apply alpha blending using NumPy
image_with_colored_people = np.copy(image)
image_with_colored_people[mask, :3] = (
    (1 - alpha_normalized[mask, None]) * image_with_colored_people[mask, :3] +
    alpha_normalized[mask, None] * color[:3]
).astype(np.uint8)

image_with_colored_people[mask, 3] = (
    alpha_channel[mask] * (1 - alpha / 255) +
    alpha * alpha_normalized[mask]
).astype(np.uint8)

# Display the original image and the image with highlighted people's bodies using OpenCV
cv2.imshow('Original Image', image)
cv2.imshow('Colored People Bodies', image_with_colored_people)
cv2.waitKey(0)
cv2.destroyAllWindows()
