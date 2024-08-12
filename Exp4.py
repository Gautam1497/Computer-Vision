import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_matching(source_image_path, reference_image_path, output_image_path):
    # Load images
    source = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    # Compute histograms and CDFs
    src_hist = np.histogram(source.flatten(), 256, [0, 256])[0]
    ref_hist = np.histogram(reference.flatten(), 256, [0, 256])[0]
    src_cdf = src_hist.cumsum() / src_hist.sum()
    ref_cdf = ref_hist.cumsum() / ref_hist.sum()

    # Create mapping function
    mapping = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)

    # Apply mapping
    matched_image = mapping[source]

    # Save and display results
    cv2.imwrite(output_image_path, matched_image)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1), plt.title("Source"), plt.imshow(source, cmap='gray')
    plt.subplot(2, 3, 2), plt.title("Reference"), plt.imshow(reference, cmap='gray')
    plt.subplot(2, 3, 3), plt.title("Matched"), plt.imshow(matched_image, cmap='gray')
    plt.subplot(2, 3, 4), plt.title("Source Hist"), plt.plot(np.histogram(source.flatten(), 256, [0, 256])[0])
    plt.subplot(2, 3, 5), plt.title("Reference Hist"), plt.plot(np.histogram(reference.flatten(), 256, [0, 256])[0])
    plt.subplot(2, 3, 6), plt.title("Matched Hist"), plt.plot(np.histogram(matched_image.flatten(), 256, [0, 256])[0])
    plt.tight_layout()
    plt.show()

# Example usage:
histogram_matching('MainAfter .jpg', 'DP2.jpg',  'lab_image.jpg')
