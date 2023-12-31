# Object Finder

## Overview

Object Finder is a Python project that uses the DeepLabV3+ model to identify people in an image and highlight them with a colored overlay. The project features a user interface (UI) created with PyQt5 for an interactive experience.

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
- The UI is built with PyQt5, providing a user-friendly interface for image manipulation.

## UI Screenshots

![Main Interface](./data/screenshots/interface.png)
*Caption: The main interface of the Object Finder application.*

![Image Selection](./data/screenshots/image_selection.png)
*Caption: Selecting an image using the UI.*

## File Structure

- `src/main.py`: The main script to run the object finder.
- `data/sample_images/`: Contains sample images for testing.
- `requirements.txt`: Lists the required Python libraries and their versions.

## Contributions

- [@Diogo Faria Martins](https://github.com/Lyne986)
- [@김서해](https://github.com/westsea6535)
- [@Bastien Boymond](https://github.com/BastienBoymond)
- [@Hugo Nini](https://github.com/Carpetic)

## Acknowledgments

- DeepLabV3+ model: [torchvision.models.segmentation.deeplabv3_resnet101](https://pytorch.org/vision/stable/models.html#deeplabv3-resnet101)
- OpenCV: [opencv-python](https://pypi.org/project/opencv-python/)

**Note**: This project adheres to specific guidelines. Numpy is intentionally excluded, and only input/output functions from OpenCV are used.

Feel free to customize the project to fit your specific needs!