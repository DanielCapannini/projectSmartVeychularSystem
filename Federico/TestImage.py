import cv2
import numpy as np
import time
import tkinter as tk
import os
print("Directory corrente:", os.getcwd())

def calculate_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

def perpendicular_distance(x1, y1, x2, y2, x0, y0):
    # Calcola la distanza perpendicolare da (x0, y0) alla linea definita da (x1, y1) e (x2, y2)
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

def are_lines_parallel(line1, line2, image, min_distance=20.0, tolerance=5.0):
    # Estrai le coordinate delle due linee
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Calcola gli angoli delle due linee
    angle1 = calculate_angle(x1, y1, x2, y2)
    angle2 = calculate_angle(x3, y3, x4, y4)
    
    # Verifica se la differenza tra gli angoli è minore della tolleranza (cioè se sono parallele)
    if abs(angle1 - angle2) < tolerance:
        # Calcola la distanza perpendicolare tra la prima linea (x1, y1, x2, y2) e un punto sulla seconda linea (x3, y3)
        distance = perpendicular_distance(x1, y1, x2, y2, x3, y3)
        
        # Se la distanza tra le due linee è inferiore alla soglia minima, non disegnare il punto
        if distance < min_distance:
            return False  # Non disegnare il punto se la distanza è troppo piccola
        
        # Calcola il punto medio tra i punti estremi delle due linee
        # Punto medio della prima linea
        center1_x = (x1 + x2) / 2
        center1_y = (y1 + y2) / 2
        
        # Punto medio della seconda linea
        center2_x = (x3 + x4) / 2
        center2_y = (y3 + y4) / 2
        
        # Calcola il punto medio tra i due punti medi delle linee
        center_x = (center1_x + center2_x) / 2
        center_y = (center1_y + center2_y) / 2
        
        # Disegna un cerchio rosso al centro
        cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        
        # Restituisce True se le linee sono parallele
        return True
    
    # Restituisce False se le linee non sono parallele
    return False


def processImage(imageURL):
    image = cv2.imread(imageURL)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

imageURL = "./Federico/img/parcheggio.jpg"
image = cv2.imread(imageURL)

# Verifica che l'immagine sia stata caricata correttamente
if image is None:
    print("Errore nel caricare l'immagine!")
    exit()

# Converti in scala di grigi
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Soglia per isolare le linee bianche
_, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Applicazione di Canny per il rilevamento dei contorni
edges = cv2.Canny(thresholded, 50, 150)

# Trova le linee usando la trasformata di Hough
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Se non troviamo linee, esci
if lines is None:
    print("Nessuna linea trovata!")
    exit()

# Vettore per memorizzare tutte le linee
all_lines = []

# Aggiungi tutte le linee all'array
for line in lines:
    x1, y1, x2, y2 = line[0]
    all_lines.append((x1, y1, x2, y2))

# Disegna tutte le linee sull'immagine
for x1, y1, x2, y2 in all_lines:
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Ciclo per confrontare tutte le coppie di linee e verificare se sono parallele
for i in range(len(all_lines)):
    for j in range(i + 1, len(all_lines)):
        line1 = all_lines[i]
        line2 = all_lines[j]
        
        # Verifica se le linee sono parallele
        if are_lines_parallel(line1, line2, image):
            print(f"La linea {i + 1} è parallela alla linea {j + 1}")

# Mostra l'immagine con le linee disegnate
cv2.imshow('Parcheggi Rilevati', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
