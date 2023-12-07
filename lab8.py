import cv2
import numpy as np
from matplotlib import pyplot as plt

imgIn = cv2.imread('ATU.jpg',)

imgGrey = cv2.cvtColor(imgIn, cv2.COLOR_BGR2GRAY)

# Create a figure to display the images
plt.figure(figsize=(12, 8))

# Create subplot: 2 rows, 2 columns
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(imgIn, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(imgGrey, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Greyscale Image')
plt.xticks([]), plt.yticks([])

'''
# Apply Gaussian Blur with different kernel sizes
kernelSize1 = (3, 3)
kernelSize2 = (13, 13)
imgBlurred1 = cv2.GaussianBlur(imgGrey, kernelSize1, 0)
imgBlurred2 = cv2.GaussianBlur(imgGrey, kernelSize2, 0)

plt.subplot(2, 2, 3)  # Bottom-left
plt.imshow(cv2.cvtColor(imgBlurred1, cv2.COLOR_BGR2RGB))
plt.title(f'Blurred with {kernelSize1} Kernel')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)  # Bottom-right
plt.imshow(cv2.cvtColor(imgBlurred2, cv2.COLOR_BGR2RGB))
plt.title(f'Blurred with {kernelSize2} Kernel')
plt.xticks([]), plt.yticks([])
'''

sobelHorizontal = cv2.Sobel(imgIn, cv2.CV_64F, 1, 0, ksize=5)  # x direction
sobelVertical = cv2.Sobel(imgIn, cv2.CV_64F, 0, 1, ksize=5)    # y direction

# Combine Sobel horizontal and vertical
combinedSobel = cv2.magnitude(sobelHorizontal, sobelVertical)

# Canny Edge Detection
cannyThreshold = 100
cannyParam2 = 300
canny = cv2.Canny(imgIn, cannyThreshold, cannyParam2)

plt.subplot(2, 3, 3)
plt.imshow(sobelHorizontal, cmap='gray')
plt.title('Sobel X')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 4)
plt.imshow(sobelVertical, cmap='gray')
plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 5)
plt.imshow(combinedSobel, cmap='gray')
plt.title('Sobel Sum')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 6)
plt.imshow(canny, cmap='gray')
plt.title('Canny Edge Detection')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
