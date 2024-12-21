import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Caricamento dell'immagine
def load_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Conversione in scala di grigi
    return image, gray

# 2. Pre-elaborazione (riduzione del rumore)
def preprocess_image(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Filtro Gaussiano per ridurre il rumore
    edges = cv2.Canny(blurred, 50, 150)          # Rilevamento dei bordi con Canny
    return edges

# 3. Rilevamento delle linee con la trasformata di Hough
def detect_lines(edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 150, 180)  # Trasformata di Hough
    return lines

# 4. Disegno delle linee sull'immagine originale
def draw_lines(image, lines):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# 5. Visualizzazione del risultato
def display_image(image, title="Image"):
    cv2.imshow(title, image)  # Mostra l'immagine in una finestra OpenCV
    cv2.waitKey(0)  # Aspetta che l'utente prema un tasto per chiudere
    cv2.destroyAllWindows()  # Chiude tutte le finestre

# 6. Main function
def main(image_path):
    image, gray = load_image(image_path)
    edges = preprocess_image(gray)
    lines = detect_lines(edges)
    result = draw_lines(image.copy(), lines)
    display_image(result, "Rilevamento Linee Parcheggio")

# Esegui l'algoritmo su un'immagine
image_path = "./Federico/img/parcheggio3.jpg" 
image_path2 = "./Federico/img/parcheggio.jpg" 
main(image_path)
main(image_path2)