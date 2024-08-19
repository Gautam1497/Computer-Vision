import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply spatial filter (Gaussian Blur)
def apply_spatial_filter(image):
    # Applying a Gaussian blur to reduce noise
    gaussian_blur = cv2.GaussianBlur(image, (7, 7), 0)
    return gaussian_blur

# Function to apply frequency domain filter (High-pass filter using Fourier Transform)
def apply_frequency_filter(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Fourier transform to the image
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Get the magnitude spectrum (for visualization)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # Creating a mask for high-pass filtering
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2

    # Create a mask with a low-frequency cutoff (high-pass filter)
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # Apply the mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back, magnitude_spectrum

# Load input image
image = cv2.imread('MainAfter .jpg')

# Apply spatial filter (Gaussian blur)
spatial_filtered_image = apply_spatial_filter(image)

# Apply frequency filter (High-pass filter)
frequency_filtered_image, magnitude_spectrum = apply_frequency_filter(image)

# Plotting the results
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.axis('off')

plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(spatial_filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Spatial Filter (Gaussian Blur)'), plt.axis('off')

plt.subplot(2, 2, 3), plt.imshow(frequency_filtered_image, cmap='gray')
plt.title('Frequency Filter (High-pass)'), plt.axis('off')

plt.subplot(2, 2, 4), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.axis('off')

plt.show()
