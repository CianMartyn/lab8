import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',)


# Load the image
imgOrig = cv2.imread('ATU.jpg',)

# Convert to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Convert original image to RGB for correct color rendering in Matplotlib
imgOrigRGB = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB)

# Plotting
plt.subplot(2, 1, 1)
plt.imshow(imgOrigRGB)
plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 1, 2)
plt.imshow(imgGray, cmap='gray')
plt.title('GrayScale')
plt.xticks([]), plt.yticks([])

plt.show()