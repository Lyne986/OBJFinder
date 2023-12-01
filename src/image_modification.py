# image_modification.py
import cv2

class ImageModification:
    @staticmethod
    def load_image(image_path):
        # Load your image without using cv2.cvtColor or NumPy
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as BGR

        # Convert the image to RGB format without using NumPy
        for y in range(len(image)):
            for x in range(len(image[0])):
                image[y][x][0], image[y][x][2] = image[y][x][2], image[y][x][0]

        return image

    @staticmethod
    def create_people_mask(segmentation, people_class_index=15):
        # Create a mask for people's bodies
        return segmentation == people_class_index

    @staticmethod
    def color_and_blur_people(image, mask, alpha, blur_level):
        # Change the color and alpha of people's bodies without using NumPy
        image_with_colored_people = []

        # Normalize alpha channel to the range [0, 1]
        alpha_normalized = alpha / 255.0

        for y, row in enumerate(image):
            new_row = []
            for x, pixel in enumerate(row):
                if mask[y][x]:
                    # Convert the list to a tuple before multiplication
                    color_tuple = (0, 0, 255)
                    updated_pixel = tuple(
                        int((1 - alpha_normalized) * pixel[c] +
                            alpha_normalized * color_tuple[c])
                        for c in range(3)
                    )
                    new_row.append(updated_pixel)
                else:
                    new_row.append(pixel)
            image_with_colored_people.append(new_row)

        # Apply blur to the detected people's region
        image_with_colored_people = ImageModification.apply_blur(image_with_colored_people, mask, blur_level)

        return image_with_colored_people

    @staticmethod
    def apply_blur(image, mask, blur_level):
        # Apply blur to the detected people's region without using NumPy
        if blur_level > 0:
            # Find the bounding box of the people's region
            min_x, min_y, max_x, max_y = ImageModification.get_people_bbox(mask)

            # Apply blur to the bounding box region
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if mask[y][x]:
                        # Blur only the people's region
                        image[y][x] = ImageModification.apply_pixel_blur(image, mask, x, y, blur_level)

        return image

    @staticmethod
    def apply_pixel_blur(image, mask, x, y, blur_level):
        # Apply blur to a single pixel using a simple averaging filter
        height, width = len(image), len(image[0])
        channels = len(image[0][0])

        total_pixels = 0
        total_color = [0] * channels

        for i in range(max(0, y - blur_level), min(height, y + blur_level + 1)):
            for j in range(max(0, x - blur_level), min(width, x + blur_level + 1)):
                if mask[i][j]:
                    total_pixels += 1
                    total_color = list(map(sum, zip(total_color, image[i][j])))

        averaged_color = [int(val / total_pixels) for val in total_color]

        return tuple(averaged_color)

    @staticmethod
    def get_people_bbox(mask):
        # Find the bounding box of the people's region
        rows = [any(mask[y]) for y in range(len(mask))]
        cols = [any(mask[y][x] for y in range(len(mask))) for x in range(len(mask[0]))]

        min_x = cols.index(True)
        max_x = len(cols) - 1 - cols[::-1].index(True)
        min_y = rows.index(True)
        max_y = len(rows) - 1 - rows[::-1].index(True)

        return min_x, min_y, max_x, max_y
