import cv2
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

def rgb_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_path = 'DP2.jpg'  
color_image = cv2.imread(image_path)

if color_image is None:
    print(f"Error: Unable to read image file {image_path}")
else:
    display_image('Original Color Image (BGR)', color_image)

    grayscale_image = rgb_to_grayscale(color_image)

    display_image('Grayscale Image', grayscale_image, cmap='gray')

    cv2.imwrite('grayscale_image.jpg', grayscale_image)

    print("Grayscale image saved successfully.")
