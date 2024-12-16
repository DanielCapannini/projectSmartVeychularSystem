import cv2
import numpy as np
import random
import os
import uuid
from typing import List, Tuple, Optional

# Funzione per migliorare l'immagine (equalizzazione dell'istogramma)
def preprocess_image(image):

    #image = cv2.imread(imageURL)
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
    # Crea una copia dell'immagine per non modificare l'originale
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Converti l'immagine in un'immagine a colori
    found_polygons = []  # Per memorizzare i vertici dei poligoni trovati

    # Trova i contorni nell'immagine binaria (invertiamo l'immagine per cercare il nero)
    contours, _ = cv2.findContours(255 - image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ottieni le dimensioni dell'immagine
    height, width = image.shape

    # Filtra i contorni per area minima e quelli che non toccano i bordi
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)  # Ottieni il rettangolo del contorno

        # Controlla che il contorno non tocchi i bordi
        if x > 0 and y > 0 and (x + w) < width and (y + h) < height:
            if area > min_area:  # Verifica se l'area del contorno è maggiore della soglia
                # Riempi il contorno con il colore specificato
                cv2.drawContours(colored_image, [contour], -1, color, thickness=cv2.FILLED)
                
                # Approssima il contorno a un poligono
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Se il poligono ha 4 lati, memorizza i suoi angoli
                if len(approx) == 4:
                    corners = [(point[0][0], point[0][1]) for point in approx]
                    found_polygons.append(corners)
                    
                    # Disegna i vertici sull'immagine
                    for corner in corners:
                        cv2.circle(colored_image, corner, radius=10, color=(0, 0, 255), thickness=-1)  # Cerchio rosso
                        
                    center = find_center_of_polygon(corners)
                    center = (int(center[0]), int(center[1]))
                    cv2.circle(colored_image, center, radius=10, color=(0, 255, 255), thickness=-1)

    # Ritorna l'immagine, il booleano e la lista dei poligoni trovati
    return colored_image, len(found_polygons) > 0, center

def find_center_of_polygon(corners):
    # Assumiamo che `corners` sia una lista di tuple con 4 punti, ad esempio [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    # Calcola la media delle coordinate x e y
    center_x = sum([corner[0] for corner in corners]) / len(corners)
    center_y = sum([corner[1] for corner in corners]) / len(corners)
    
    return (center_x, center_y)

def process_image(imageURL, i):
    # Assicurati che la cartella 'output' esista
    output_dir = "./Federico/project/output"
    os.makedirs(output_dir, exist_ok=True)  # Crea la cartella se non esiste

    # Preprocessing dell'immagine
    image_opened = preprocess_image(imageURL)  # Pre-processing dell'immagine
    image_with_lines = draw_horizontal_line(image_opened)  # Rilevamento delle linee
    image_parking_found, parking_exist, center = color_enclosed_black_areas(image_with_lines)

    if parking_exist:
        print(f"Centro rilevato per il parcheggio: {center}")

    # Mostra l'immagine con la linea
    #cv2.imshow("Image with Horizontal Lines", image_parking_found)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Salva l'immagine risultante nella cartella 'output'
    unique_filename = f"./output/image_" + str(i) + ".png"    
    #image_parking_found.save_to_disk(f'output/{unique_filename.frame}.png')
    cv2.imwrite(unique_filename, image_parking_found)

    return parking_exist, center

"""
# Esegui la funzione per le immagini
imageURL = "./Federico/img/parcheggio.jpg"
imageURLAlto = "./Federico/img/parcheggioAlto.jpg"
imageURL2 = "./Federico/img/parcheggio3.jpg"
ImageURLNew = "./RL/guida/output3/4725.png"
ImageURLNew2 = "./RL/guida/output3/4700.png"
ImageURLNew3 = "./RL/guida/output3/4790.png"
ImageURLNew4 = "./RL/guida/output3/4800.png"

process_image(ImageURLNew)
process_image(ImageURLNew2)
process_image(ImageURLNew3)
process_image(ImageURLNew4)
"""
