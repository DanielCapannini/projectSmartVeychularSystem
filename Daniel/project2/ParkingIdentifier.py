import cv2
import numpy as np
import random
import os
import uuid
from typing import List, Tuple, Optional

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return mask_cleaned

def draw_horizontal_line(image):
    image_with_line = image.copy()
    height, width = image.shape
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = []
    min_contour_area = 500
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            large_contours.append(contour)
    lowerY = 0
    for x in range(width):
        for y in range(height-1, -1, -1):
            if image[y, x] == 255: 
                for contour in large_contours:
                    if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                        if lowerY < y:
                            lowerY = y
                        break 
                break  
    cv2.line(image_with_line, (0, lowerY), (width-1, lowerY), (255, 255, 255), thickness=5)  # Linea bianca spessa
    return image_with_line

def color_enclosed_black_areas(image, color=(0, 255, 0), min_area=500, epsilon_factor=0.02):
    found_polygons = []
    contours, _ = cv2.findContours(255 - image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape
    center = None
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour) 
        if x > 0 and y > 0 and (x + w) < width and (y + h) < height:
            if area > min_area:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    corners = [(point[0][0], point[0][1]) for point in approx]
                    found_polygons.append(corners)
                    center = find_center_of_polygon(corners)
                    center = (int(center[0]), int(center[1]))
                    center = int(center[0]) - (width/2)
    return len(found_polygons) > 0, center

def find_center_of_polygon(corners):
    center_x = sum([corner[0] for corner in corners]) / len(corners)
    center_y = sum([corner[1] for corner in corners]) / len(corners)
    return (center_x, center_y)

def process_image(imageURL, i):
    image_opened = preprocess_image(imageURL)  # Pre-processing dell'immagine
    image_with_lines = draw_horizontal_line(image_opened)  # Rilevamento delle linee
    parking_exist, center = color_enclosed_black_areas(image_with_lines)
    return parking_exist, center