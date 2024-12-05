import cv2
import numpy as np
import random

def generate_shares(image_path, output_share1_path, output_share2_path):
    # Load the binary image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

    # Define sub-pixel patterns for (2, 2) scheme
    C0 = [
        [[1, 1, 0, 0], [1, 0, 1, 0]],
        [[1, 0, 0, 1], [0, 1, 1, 0]],
        [[0, 1, 1, 0], [1, 0, 0, 1]],
        [[0, 0, 1, 1], [1, 1, 0, 0]],
    ]
    C1 = [
        [[1, 1, 0, 0], [0, 0, 1, 1]],
        [[1, 0, 1, 0], [0, 1, 0, 1]],
        [[0, 1, 1, 0], [1, 0, 0, 1]],
        [[0, 0, 1, 1], [1, 1, 0, 0]],
    ]

    # Get dimensions
    height, width = binary_img.shape

    # Initialize empty arrays for shares
    share1 = np.zeros((height * 2, width * 2), dtype=np.uint8)
    share2 = np.zeros((height * 2, width * 2), dtype=np.uint8)

    # Generate a random sequence number p
    p = random.randint(100, 1000)

    # Create the shares
    for i in range(height):
        for j in range(width):
            pixel = binary_img[i, j]  # 0 for black, 1 for white
            rand_index = (i * width + j) % p % 4  # Modulo operation to get index
            if pixel == 0:  # Black pixel
                subpixels = C0[rand_index]
            else:  # White pixel
                subpixels = C1[rand_index]

            # Fill in the shares with sub-pixels
            share1[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = np.array(subpixels[0]).reshape(2, 2) * 255
            share2[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = np.array(subpixels[1]).reshape(2, 2) * 255

    # Save the shares as images
    cv2.imwrite(output_share1_path, share1)
    cv2.imwrite(output_share2_path, share2)
    print(f"Shares saved as {output_share1_path} and {output_share2_path}")

# Usage example
input_image_path = "./Input.png"  # Replace with your input file path
output_share1_path = "share1.png"
output_share2_path = "share2.png"

generate_shares(input_image_path, output_share1_path, output_share2_path)