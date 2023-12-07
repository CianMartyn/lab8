import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',)

# Load the image
imgOrig = cv2.imread('ATU.jpg',)

imgGrey = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur with different kernel sizes
kernelSize1 = (3, 3)
kernelSize2 = (13, 13)
imgBlurred1 = cv2.GaussianBlur(imgGrey, kernelSize1, 0)
imgBlurred2 = cv2.GaussianBlur(imgGrey, kernelSize2, 0)

# Create a figure to display the images
plt.figure(figsize=(10, 8))

# Create subplot: 2 rows, 2 columns
plt.subplot(2, 2, 1)  # Top-left
plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)  # Top-right
plt.imshow(cv2.cvtColor(imgGrey, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Greyscale Image')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3)  # Bottom-left
plt.imshow(cv2.cvtColor(imgBlurred1, cv2.COLOR_BGR2RGB))
plt.title(f'Blurred with {kernelSize1} Kernel')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)  # Bottom-right
plt.imshow(cv2.cvtColor(imgBlurred2, cv2.COLOR_BGR2RGB))
plt.title(f'Blurred with {kernelSize2} Kernel')
plt.xticks([]), plt.yticks([])

# Display the plot
plt.tight_layout()
plt.show()