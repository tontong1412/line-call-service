import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread("court.jpg")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

# Calculate gradients along the X and Y axis
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

# Combine the two gradients
sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = np.uint8(sobel_combined)

# Apply Canny edge detector
canny_edges = cv2.Canny(blurred, 10, 60)

# Set up the matplotlib figure
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Canny Edge Detection
plt.subplot(1, 2, 2)
plt.imshow(canny_edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

# Show the plots
plt.tight_layout()
plt.savefig("line_detection.png")  # Save as image file
print("Plot saved as plot.png")
