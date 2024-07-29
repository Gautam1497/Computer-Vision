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

image_path = 'DP2.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to read image file {image_path}")
else:
    display_image('Original Image (BGR)', image)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image('RGB Image', rgb_image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image('Grayscale Image', gray_image, cmap='gray')

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    display_image('HSV Image', hsv_image)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    display_image('Lab Image', lab_image)

    cv2.imwrite('rgb_image.jpg', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite('gray_image.jpg', gray_image)
    cv2.imwrite('hsv_image.jpg', hsv_image)
    cv2.imwrite('lab_image.jpg', lab_image)

    print("Processed images saved successfully.")
