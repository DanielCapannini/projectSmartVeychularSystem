import cv2
import numpy as np
import time
import tkinter as tk
import os
print("Directory corrente:", os.getcwd())


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
processImage(imageURL)
image = cv2.imread(imageURL)

# Converti in scala di grigi
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Soglia per isolare le linee bianche
_, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Applicazione di Canny per il rilevamento dei contorni
edges = cv2.Canny(thresholded, 100, 200)

# Trova le linee usando la trasformata di Hough
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Se non troviamo linee, esci
if lines is None:
    print("Nessuna linea trovata!")
    exit()

# Lista per memorizzare le linee orizzontali, verticali e oblique
horizontal_lines = []
vertical_lines = []
oblique_lines = []

# Funzione per calcolare l'angolo di inclinazione di una linea
def calculate_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

# Analizza le linee e separa quelle orizzontali, verticali e oblique
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = calculate_angle(x1, y1, x2, y2)

    # Considera linee orizzontali (angolo tra -10 e 10 gradi)
    if -10 < angle < 10:
        horizontal_lines.append(line)
    # Considera linee verticali (angolo tra 80 e 100 gradi o tra -100 e -80 gradi)
    elif 80 < angle < 100 or -100 < angle < -80:
        vertical_lines.append(line)
    # Altrimenti considera linee oblique
    else:
        oblique_lines.append(line)

# Ora vogliamo trovare linee che possono formare un rettangolo, quindi:
# - Almeno 2 linee parallele (orizzontali o verticali)
# - Una linea che completi il rettangolo, possibilmente obliqua

# Disegna le linee rilevate
for line in horizontal_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

for line in vertical_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

for line in oblique_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Mostra l'immagine con le linee disegnate
cv2.imshow('Rilevamento Parcheggio', image)
cv2.waitKey(0)
cv2.destroyAllWindows()