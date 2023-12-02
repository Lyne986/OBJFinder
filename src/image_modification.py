import cv2

class ImageModification:
    def __init__(self):
        self.last_modified_image = None

    def load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 

        for y in range(len(image)):
            for x in range(len(image[0])):
                image[y][x][0], image[y][x][2] = image[y][x][2], image[y][x][0]
        self.last_modified_image = image
        return image

    def create_people_mask(self, segmentation, people_class_index=15):
        return segmentation == people_class_index

    def color_people(self, image, mask, alpha, red, green, blue):
        image_with_colored_people = []

        alpha_normalized = alpha / 255.0

        for y, row in enumerate(image):
            new_row = []
            for x, pixel in enumerate(row):
                if mask[y][x]:
                    color_tuple = (red, green, blue)
                    updated_pixel = tuple(
                        int((1 - alpha_normalized) * pixel[c] +
                            alpha_normalized * color_tuple[c])
                        for c in range(3)
                    )
                    new_row.append(updated_pixel)
                else:
                    new_row.append(pixel)
            image_with_colored_people.append(new_row)

        self.last_modified_image = image_with_colored_people

        return image_with_colored_people

    def apply_blur(self, image, mask, blur_level):
        if blur_level > 0:
            min_x, min_y, max_x, max_y = self.get_people_bbox(mask)

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if mask[y][x]:
                        image[y][x] = self.apply_pixel_blur(image, mask, x, y, blur_level)
        
        self.last_modified_image = image

        return image

    def apply_pixel_blur(self, image, mask, x, y, blur_level):
        height, width = len(image), len(image[0])
        channels = len(image[0][0])

        total_pixels = 0
        total_color = [0] * channels

        for i in range(max(0, y - blur_level), min(height, y + blur_level + 1)):
            for j in range(max(0, x - blur_level), min(width, x + blur_level + 1)):
                if mask[i][j]:
                    total_pixels += 1
                    total_color[0] += image[i][j][0] 
                    total_color[1] += image[i][j][1] 
                    total_color[2] += image[i][j][2] 

        averaged_color = (
            int(total_color[0] / total_pixels),
            int(total_color[1] / total_pixels),
            int(total_color[2] / total_pixels)
        )

        return averaged_color

    def get_people_bbox(self, mask):
        rows = [any(mask[y]) for y in range(len(mask))]
        cols = [any(mask[y][x] for y in range(len(mask))) for x in range(len(mask[0]))]

        min_x = cols.index(True)
        max_x = len(cols) - 1 - cols[::-1].index(True)
        min_y = rows.index(True)
        max_y = len(rows) - 1 - rows[::-1].index(True)

        return min_x, min_y, max_x, max_y
