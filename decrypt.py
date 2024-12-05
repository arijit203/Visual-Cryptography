import cv2
import numpy as np

def reconstruct_image(share1_path, share2_path, output_image_path):
    # Load the two shares
    share1 = cv2.imread(share1_path, cv2.IMREAD_GRAYSCALE)
    share2 = cv2.imread(share2_path, cv2.IMREAD_GRAYSCALE)

    # Ensure both shares have the same dimensions
    if share1.shape != share2.shape:
        raise ValueError("Shares must have the same dimensions")

    # Get dimensions of the shares
    height, width = share1.shape

    # Initialize the reconstructed binary image
    original_height, original_width = height // 2, width // 2
    reconstructed_img = np.zeros((original_height, original_width), dtype=np.uint8)

    # Reconstruct the original image
    for i in range(original_height):
        for j in range(original_width):
            # Extract 2x2 subpixels from both shares
            subpixels_share1 = share1[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]
            subpixels_share2 = share2[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]

            # Combine the shares (stacking)
            stacked_subpixels = subpixels_share1 + subpixels_share2

            # Determine the original pixel value
            # If any subpixel is black (0 after stacking), the original pixel is black
            # Otherwise, it's white
            if np.any(stacked_subpixels == 0):
                reconstructed_img[i, j] = 0  # Black pixel
            else:
                reconstructed_img[i, j] = 255  # White pixel

    # Save the reconstructed image
    cv2.imwrite(output_image_path, reconstructed_img)
    print(f"Reconstructed image saved as {output_image_path}")

# Usage example
share1_path = "share1.png"
share2_path = "share2.png"
output_image_path = "reconstructed_image.png"

reconstruct_image(share1_path, share2_path, output_image_path)