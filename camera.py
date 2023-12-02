import cv2
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms import functional as F
import numpy as np

model = deeplabv3_resnet101(pretrained=True)
model.eval()

cap = cv2.VideoCapture(0)

cv2.waitKey(1000)

new_width, new_height = 640, 480
skip_frames = 2
frame_count = 0

while True:
    ret, frame = cap.read()

    if frame_count % skip_frames == 0:
        frame = cv2.resize(frame, (new_width, new_height))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0)

        with torch.no_grad():
            output = model(frame_tensor)['out']

        segmentation = output.argmax(1).squeeze().detach().cpu().numpy()

        people_class_index = 15  

        people_mask = segmentation == people_class_index

        blue_color = np.array([255, 0, 0], dtype=np.uint8)
        frame[people_mask] = frame[people_mask] * 0.7 + blue_color * 0.3

        cv2.imshow('Real-Time Segmentation', frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
