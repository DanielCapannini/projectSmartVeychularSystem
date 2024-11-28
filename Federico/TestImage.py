import cv2
import numpy as np
import time
import tkinter as tk
import os
print("Directory corrente:", os.getcwd())

def calculate_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

def perpendicular_distance(x1, y1, x2, y2, x0, y0):
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def are_lines_parallel(line1, line2, image, min_distance=20.0, tolerance=5.0):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    angle1 = calculate_angle(x1, y1, x2, y2)
    angle2 = calculate_angle(x3, y3, x4, y4)
    
    if abs(angle1 - angle2) < tolerance:
        distance = perpendicular_distance(x1, y1, x2, y2, x3, y3)
        
        if distance < min_distance:
            return False 
        
        center1_x = (x1 + x2) / 2
        center1_y = (y1 + y2) / 2
        center2_x = (x3 + x4) / 2
        center2_y = (y3 + y4) / 2
        center_x = (center1_x + center2_x) / 2
        center_y = (center1_y + center2_y) / 2
        
        cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        
        return True
    
    return False

def extend_line_to_connect(line1, line2, image, threshold=50.0, angle_tolerance=5.0):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    distance_1 = calculate_distance(x1, y1, x3, y3)
    distance_2 = calculate_distance(x1, y1, x4, y4)
    distance_3 = calculate_distance(x2, y2, x3, y3)
    distance_4 = calculate_distance(x2, y2, x4, y4)

    min_distance = min(distance_1, distance_2, distance_3, distance_4)

    angle1 = calculate_angle(x1, y1, x2, y2)
    angle2 = calculate_angle(x3, y3, x4, y4)

    if abs(angle1 - angle2) < angle_tolerance and min_distance < threshold:
        if min_distance == distance_1 or min_distance == distance_2:
            direction_x = x2 - x1
            direction_y = y2 - y1
            extension_length = 100  # Lunghezza dell'estensione
            x_extended = x2 + direction_x * extension_length / calculate_distance(x1, y1, x2, y2)
            y_extended = y2 + direction_y * extension_length / calculate_distance(x1, y1, x2, y2)
            line1 = (x1, y1, x_extended, y_extended)
        else:
            direction_x = x4 - x3
            direction_y = y4 - y3
            extension_length = 100  # Lunghezza dell'estensione
            x_extended = x4 + direction_x * extension_length / calculate_distance(x3, y3, x4, y4)
            y_extended = y4 + direction_y * extension_length / calculate_distance(x3, y3, x4, y4)
            line2 = (x3, y3, x_extended, y_extended)

        cv2.line(image, (int(x1), int(y1)), (int(x_extended), int(y_extended)), (0, 0, 255), 2)
        
        return True, line1 if min_distance == distance_1 or min_distance == distance_2 else line2  # Restituisci la linea aggiornata
    else:
        return False, line1  # Le linee non sono state unite, restituisci la linea originale


def processImage(imageURL):
    image = cv2.imread(imageURL)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold automatico per estrarre il bianco
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Estrazione dei bianchi con soglia personalizzata
    white_pixels = np.where(thresholded == 255)
    sorted_pixels = np.sort(gray_image[white_pixels])
    threshold_value = sorted_pixels[int(0.85 * len(sorted_pixels))]
    _, custom_thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Creazione della maschera per le aree bianche
    mask = np.zeros_like(gray_image)
    mask[custom_thresholded == 255] = gray_image[custom_thresholded == 255]
    
    # Operazioni morfologiche per migliorare l'immagine
    kernel = np.ones((3, 3), np.uint8)
    image_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    image_opened = cv2.morphologyEx(image_closed, cv2.MORPH_OPEN, (5,5))
    image_opened[image_opened > 0] = 255
    
    # Visualizza l'immagine con le aree bianche isolate
    cv2.imshow("Thresholded Image", image_opened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Applicazione della rilevazione delle linee (Hough Transform)
    detect_lines(image_opened)

# Funzione per rilevare le linee in un'immagine usando la trasformata di Hough
def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold per ottenere un'immagine binaria con le linee
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Rilevazione dei bordi con Canny
    edges = cv2.Canny(thresholded, 50, 150)

    # Trova le linee usando la trasformata di Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        print("Nessuna linea trovata!")
        return

    all_lines = []

    # Aggiungi tutte le linee trovate
    for line in lines:
        x1, y1, x2, y2 = line[0]
        all_lines.append((x1, y1, x2, y2))

    # Disegnare tutte le linee sull'immagine originale
    image = cv2.imread(imageURL)
    for x1, y1, x2, y2 in all_lines:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Unire linee se sono collegate
    for i in range(len(all_lines)):
        for j in range(i + 1, len(all_lines)):
            line1 = all_lines[i]
            line2 = all_lines[j]

            if extend_line_to_connect(line1, line2, image):
                print(f"Le linee {i + 1} e {j + 1} sono state unite.")

            if are_lines_parallel(line1, line2):
                print(f"La linea {i + 1} è parallela alla linea {j + 1}")

    # Mostra l'immagine con le linee
    cv2.imshow("Detected Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



imageURL = "./Federico/img/parcheggio.jpg"
imageURL2 = "./Federico/img/parcheggio3.jpg"

processImage(imageURL)
processImage(imageURL2)
#image = cv2.imread(imageURL)
"""
if image is None:
    print("Errore nel caricare l'immagine!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(thresholded, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

if lines is None:
    print("Nessuna linea trovata!")
    exit()

all_lines = []

# Aggiungi tutte le linee all'array
for line in lines:
    x1, y1, x2, y2 = line[0]
    all_lines.append((x1, y1, x2, y2))

for x1, y1, x2, y2 in all_lines:

for i in range(len(all_lines)):
    for j in range(i + 1, len(all_lines)):
        line1 = all_lines[i]
        line2 = all_lines[j]

        if extend_line_to_connect(line1, line2, image):
            print(f"Le linee {i + 1} e {j + 1} sono state unite.")

        if are_lines_parallel(line1, line2, image):
            print(f"La linea {i + 1} è parallela alla linea {j + 1}")
"""
cv2.imshow('Parcheggi Rilevati', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
