import cv2
import numpy as np

def apply_rectangular_black_mask_center(image_path, output_path, mask_ratio=0.7):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")

    height, width = image.shape[:2]

    # Calculate mask size to cover approximately mask_ratio of image area
    mask_area = int(height * width * mask_ratio)

    # Approximate dimensions of the rectangle (try to make it near square)
    mask_height = int(np.sqrt(mask_area * height / width))
    mask_width = int(np.sqrt(mask_area * width / height))

    # Calculate top-left corner to center the rectangle
    top_left_x = (width - mask_width) // 2
    top_left_y = (height - mask_height) // 2

    # Copy image to avoid modifying original
    masked_image = image.copy()

    # Draw a filled black rectangle on the image (BGR color (0,0,0))
    cv2.rectangle(masked_image, (top_left_x, top_left_y),
                  (top_left_x + mask_width, top_left_y + mask_height),
                  (0, 0, 0), thickness=-1)

    # Save the result
    cv2.imwrite(output_path, masked_image)

# Example usage
apply_rectangular_black_mask_center('IMG_20251016_125132_cropped.png', 'output_rectangular_mask_center.jpg', mask_ratio=0.20)
