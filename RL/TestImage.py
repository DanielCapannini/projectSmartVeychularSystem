import cv2
import numpy as np
import os
import tkinter as tk

image = cv2.imread("../33271.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
x_start, y_start = 0, 240  # Top-left corner
x_end, y_end = 600, 800      # Bottom-right corner
gray_image = gray_image[y_start:y_end, x_start:x_end]
_, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
white_pixels = np.where(thresholded == 255)
sorted_pixels = np.sort(gray_image[white_pixels])
threshold_value = sorted_pixels[int(0.85 * len(sorted_pixels))]
_, custom_thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
mask = np.zeros_like(gray_image)
mask[custom_thresholded == 255] = gray_image[custom_thresholded == 255]
kernel = np.ones((3, 3), np.uint8)
image_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
image_opened = cv2.morphologyEx(image_closed, cv2.MORPH_OPEN, (5,5))
image_opened[image_opened > 0] = 255
cv2.imshow("Result Image", image_opened)
cv2.waitKey(0)
cv2.destroyAllWindows()