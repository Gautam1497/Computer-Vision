import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image, cmap=None):
    plt.figure(figsize=(8, 8))
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

image_path = 'DP2.jpg' 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Unable to read image file {image_path}")
else:
    display_image('Original Image', image, cmap='gray')

    negative_image = 255 - image
    display_image('Negative Transformation', negative_image, cmap='gray')
    cv2.imwrite('negative_image.jpg', negative_image)

    c = 255 / np.log(1 + np.max(image)) 
    log_image = c * np.log(1 + image.astype(float)) 
    log_image = np.clip(log_image, 0, 255)  
    log_image = np.array(log_image, dtype=np.uint8)
    display_image('Log Transformation', log_image, cmap='gray')
    cv2.imwrite('log_image.jpg', log_image)

    gamma = 0.4 
    gamma_image = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)
    display_image(f'Gamma Transformation (Gamma={gamma})', gamma_image, cmap='gray')
    cv2.imwrite('gamma_image.jpg', gamma_image)

    hist_eq_image = cv2.equalizeHist(image)
    display_image('Histogram Equalization', hist_eq_image, cmap='gray')
    cv2.imwrite('hist_eq_image.jpg', hist_eq_image)

    print("Enhanced images saved successfully.")
