import cv2
import numpy as np
import random
from typing import List, Tuple, Optional

# Funzione per migliorare l'immagine (equalizzazione dell'istogramma)
def preprocess_image(imageURL):

    image = cv2.imread(imageURL)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalizzazione dell'istogramma per migliorare il contrasto
    gray_image = cv2.equalizeHist(gray_image)

    # Threshold automatico per estrarre il bianco
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Estrazione dei bianchi con soglia personalizzata
    white_pixels = np.where(thresholded == 255)
    sorted_pixels = np.sort(gray_image[white_pixels])
    threshold_value = sorted_pixels[int(0.85 * len(sorted_pixels))]
    _, custom_thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Creazione della maschera per le aree bianche
    mask = np.zeros_like(gray_image)
    mask[custom_thresholded == 255] = 255  # Aree bianche
    mask[custom_thresholded != 255] = 0   # Aree non bianche (nero)

    # Operazioni morfologiche per migliorare l'immagine
    kernel = np.ones((5, 5), np.uint8)
    image_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    image_opened = cv2.morphologyEx(image_closed, cv2.MORPH_OPEN, (5,5))
    image_opened[image_opened > 0] = 255

    return image_opened

def draw_horizontal_line(image):
    """
    Trova il punto più basso di un pixel bianco in ogni colonna e disegna una linea orizzontale
    bianca spessa su quel punto per ogni colonna, ma solo se il pixel bianco fa parte di un
    contorno abbastanza grande.
    
    Args:
        image (numpy.ndarray): Immagine binaria (0 e 255).
    
    Returns:
        numpy.ndarray: Immagine con la linea orizzontale disegnata.
    """
    # Crea una copia dell'immagine per disegnare la linea
    image_with_line = image.copy()

    # Ottieni le dimensioni dell'immagine
    height, width = image.shape

    # Trova i contorni nell'immagine
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calcola l'area di ogni contorno e memorizza quelli abbastanza grandi
    large_contours = []
    min_contour_area = 500  # Modifica questa soglia in base alle tue esigenze
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            large_contours.append(contour)

    # Scorri ogni colonna da sotto verso l'alto per trovare il pixel bianco più basso
    lowerY = 0
    for x in range(width):
        for y in range(height-1, -1, -1):
            if image[y, x] == 255:  # Trova il primo pixel bianco più basso
                # Verifica se il pixel fa parte di un contorno abbastanza grande
                for contour in large_contours:
                    if cv2.pointPolygonTest(contour, (x, y), False) >= 0:  # Il punto (x, y) è dentro il contorno
                        if lowerY < y:
                            lowerY = y
                        break  # Fermati appena trovi il contorno valido

                break  # Fermati quando trovi il primo pixel bianco per colonna

    # Disegna la linea orizzontale sulla riga più bassa trovata
    cv2.line(image_with_line, (0, lowerY), (width-1, lowerY), (255, 255, 255), thickness=5)  # Linea bianca spessa

    return image_with_line

def color_shapes(image, color=(0, 255, 0), min_area=500):
    """
    Trova le forme nell'immagine binaria (dentro le linee bianche) e le colora con un altro colore.
    
    Args:
        image (numpy.ndarray): Immagine binaria (0 e 255).
        color (tuple): Colore in formato BGR (default verde).
        min_area (int): L'area minima di un contorno per essere colorato (default 500).
    
    Returns:
        numpy.ndarray: Immagine con le forme colorate.
    """
    # Crea una copia dell'immagine per non modificare l'originale
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Converti l'immagine in un'immagine a colori

    # Trova i contorni nell'immagine binaria
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra i contorni per area minima
    for contour in contours:
        if cv2.contourArea(contour) > min_area:  # Verifica se l'area del contorno è maggiore della soglia
            # Riempi il contorno con il colore specificato
            cv2.drawContours(colored_image, [contour], -1, color, thickness=cv2.FILLED)

    return colored_image

def process_image(imageURL):
    image_opened = preprocess_image(imageURL)  # Pre-processing dell'immagine
    image_with_lines = draw_horizontal_line(image_opened)  # Rilevamento delle linee
    image_parking_found = color_shapes(image_with_lines)

    # Mostra l'immagine con la linea
    cv2.imshow("Image with Horizontal Lines", image_parking_found)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Salva l'immagine risultante
    cv2.imwrite("image_with_horizontal_lines.png", image_parking_found)

# Esegui la funzione per le immagini
imageURL = "./Federico/img/parcheggio.jpg"
imageURLAlto = "./Federico/img/parcheggioAlto.jpg"
imageURL2 = "./Federico/img/parcheggio3.jpg"
ImageURLNew = "./RL/guida/output3/4365.png"
ImageURLNew2 = "./RL/guida/output3/4700.png"

process_image(ImageURLNew)
process_image(ImageURLNew2)
