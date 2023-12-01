```markdown
# Object Finder

## Overview

Object Finder is a Python project that uses the DeepLabV3+ model to identify people in an image and highlight them with a colored overlay.

## Requirements

- Python 3.x
- Install the required libraries using the following command:

  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ObjectFinder.git
   cd ObjectFinder
   ```

2. Run the project using the provided script:

   ```bash
   python3 ./src/main.py
   ```

## Details

- The project uses the DeepLabV3+ model from torchvision to perform semantic segmentation on the input image.
- Identified people are highlighted with a specified color and transparency level.
- OpenCV is utilized for image processing.

## File Structure

- `src/main.py`: The main script to run the object finder.
- `data/sample_images/`: Contains sample images for testing.
- `requirements.txt`: Lists the required Python libraries and their versions.

## Contributions

- Developer: [@Diogo Faria Martins](https://github.com/Lyne986)
- Developer: [@김서해](https://github.com/westsea6535)
- Developer: [@Bastien Boymond](https://github.com/BastienBoymond)
- Developer: [@Hugo Nini](https://github.com/Carpetic)

## Acknowledgments

- DeepLabV3+ model: [torchvision.models.segmentation.deeplabv3_resnet101](https://pytorch.org/vision/stable/models.html#deeplabv3-resnet101)
- OpenCV: [opencv-python](https://pypi.org/project/opencv-python/)

Feel free to customize the project to fit your specific needs!
```