import cv2
import numpy as np
import math
from typing import List, Tuple, Optional
from Utility import preprocess_image

def find_and_draw_longest_white_line(binary_image):
    """
    Trova e disegna la linea bianca più lunga in un'immagine binaria, indipendentemente dall'inclinazione.

    Args:
        image (numpy.ndarray): Immagine binaria (0 e 255).

    Returns:
        numpy.ndarray: Immagine con la linea più lunga disegnata.
    """
    image_with_line = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    longest_line = None
    max_length = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon=1, closed=False)
        for i in range(len(approx) - 1):
            x1, y1 = approx[i][0]
            x2, y2 = approx[i + 1][0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            lengthX = abs(x2-x1)
            lengthY = abs(y2-y1)
            if lengthX > lengthY*2:
                if lengthX > max_length:
                    max_length = lengthX
                    longest_line = ((x1, y1), (x2, y2))
    if longest_line is not None:
        cv2.line(image_with_line, longest_line[0], longest_line[1], (0, 0, 255), 2)
        ((x1, y1), (x2, y2)) = longest_line
        angle_radians = np.arctan2(y2 - y1, x2 - x1)
    angle_radians = -(angle_radians) + (angle_radians*0.015) 
    return angle_radians

def draw_line_at_angle_to_two_points(image, angle_radians):
    """
    Trova il punto più basso di un pixel bianco in ogni colonna e disegna una linea inclinata
    di un angolo specifico rispetto all'orizzontale a partire da quel punto.

    Args:
        image (numpy.ndarray): Immagine binaria (0 e 255).
        angle_radians (float): Angolo della linea in radianti.

    Returns:
        numpy.ndarray: Immagine con la linea inclinata disegnata.
    """
    image_with_line = image.copy()
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 100]
    if not large_contours:
        return image_with_line  
    lowerY, xPoint = -1, -1
    lowerY2, xPoint2 = -1, -1
    min_distance = 200
    for contour in large_contours:
        for point in contour:
            x, y = point[0]
            if y > lowerY or (y == lowerY and abs(x - xPoint) > min_distance):
                lowerY2, xPoint2 = lowerY, xPoint
                lowerY, xPoint = y, x
            elif y > lowerY2 and abs(x - xPoint) > min_distance:
                lowerY2, xPoint2 = y, x
    if lowerY == -1 or lowerY2 == -1:
        return image_with_line  
    if abs(angle_radians) >= 0.5:
        lowerY -= 20
    start_x, start_y = xPoint, lowerY
    dx = xPoint2 - start_x
    dy = lowerY2 - start_y
    angle_radians = math.atan2(dy, dx)
    extension_length = 500
    extended_start_x = int(start_x - extension_length * math.cos(angle_radians))
    extended_start_y = int(start_y - extension_length * math.sin(angle_radians))
    extended_end_x = int(xPoint2 + extension_length * math.cos(angle_radians))
    extended_end_y = int(lowerY2 + extension_length * math.sin(angle_radians))
    cv2.line( image_with_line, (extended_start_x, extended_start_y), (extended_end_x, extended_end_y), (255, 255, 255), thickness=30)
    return image_with_line

def color_enclosed_black_areas(image, color=(0, 255, 0), min_area=500, epsilon_factor=0.02):
    """
    Trova le aree nere chiuse delimitate da righe bianche, le colora e identifica i poligoni con 4 lati.
    
    Args:
        image (numpy.ndarray): Immagine binaria (0 e 255).
        color (tuple): Colore in formato BGR (default verde).
        min_area (int): L'area minima di un contorno per essere colorato (default 500).
        epsilon_factor (float): Fattore per approssimare il contorno.

    Returns:
        Tuple[numpy.ndarray, bool, List[List[Tuple[int, int]]]]:
            - Immagine con i poligoni colorati.
            - Booleano che indica se è stato trovato almeno un poligono con 4 lati.
            - Lista di liste dei vertici dei poligoni trovati.
    """
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
    found_polygons = [] 
    contours, _ = cv2.findContours(255 - image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape
    center = None
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)  
        if x > 0 and y > 0 and (x + w) < width and (y + h) < height:
            if area > min_area: 
                cv2.drawContours(colored_image, [contour], -1, color, thickness=cv2.FILLED)
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    corners = [(point[0][0], point[0][1]) for point in approx]
                    found_polygons.append(corners)
                    center = find_center_of_polygon(corners)
                    center = (int(center[0]), int(center[1]))
                    center = (int(center[0]) - (width/2), int(center[1]))
    return colored_image, len(found_polygons) > 0, center

def find_center_of_polygon(corners):
    center_x = sum([corner[0] for corner in corners]) / len(corners)
    center_y = sum([corner[1] for corner in corners]) / len(corners)
    return (center_x, center_y)

def process_image(imageURL, i=0):
    image_opened = preprocess_image(imageURL) 
    angle_radians = (find_and_draw_longest_white_line(image_opened))
    image_with_lines = draw_line_at_angle_to_two_points(image_opened, angle_radians)
    _, parking_exist, center = color_enclosed_black_areas(image_with_lines)
    return parking_exist, center
