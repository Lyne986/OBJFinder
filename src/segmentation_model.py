# segmentation_model.py
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms import functional as F

#Made by 휴고 니니 / Hugo Nini #5023194
class PeopleSegmentationModel:
    def __init__(self):
        # Load the DeepLabV3+ model
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.eval()

    def apply_segmentation(self, image):
        image_tensor = F.to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image_tensor)['out']

        segmentation = output.argmax(1).squeeze().detach().cpu().numpy()
        return segmentation
