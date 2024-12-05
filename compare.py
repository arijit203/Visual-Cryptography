import cv2
import numpy as np

def calculate_psnr(original_image_path, reconstructed_image_path):
    # Load the original and reconstructed images
    original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    reconstructed = cv2.imread(reconstructed_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure both images have the same dimensions
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed images must have the same dimensions")

    # Compute Mean Squared Error (MSE)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:  # If MSE is zero, PSNR is infinite
        return float('inf')

    # Compute PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_normalized_correlation(original_image_path, reconstructed_image_path):
    # Load the original and reconstructed images
    original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    reconstructed = cv2.imread(reconstructed_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure both images have the same dimensions
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed images must have the same dimensions")

    # Flatten images to 1D arrays for correlation calculation
    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()

    # Compute normalized correlation
    numerator = np.sum(original_flat * reconstructed_flat)
    denominator = np.sqrt(np.sum(original_flat ** 2) * np.sum(reconstructed_flat ** 2))
    correlation = numerator / denominator if denominator != 0 else 0
    return correlation

# Usage example
original_image_path = "Input.png"  # Replace with the original image path
reconstructed_image_path = "reconstructed_image.png"  # Replace with the reconstructed image path

psnr = calculate_psnr(original_image_path, reconstructed_image_path)
correlation = calculate_normalized_correlation(original_image_path, reconstructed_image_path)

print(f"PSNR: {psnr:.2f} dB")
print(f"Normalized Correlation: {correlation:.4f}")