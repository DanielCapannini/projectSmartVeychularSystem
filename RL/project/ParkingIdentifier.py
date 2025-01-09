import cv2
import numpy as np
import random
import os
import uuid
import math
from typing import List, Tuple, Optional

# Funzione per migliorare l'immagine (equalizzazione dell'istogramma)
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return mask_cleaned

def find_and_draw_longest_white_line(binary_image):
    """
    Trova e disegna la linea bianca più lunga in un'immagine binaria, indipendentemente dall'inclinazione.

    Args:
        image (numpy.ndarray): Immagine binaria (0 e 255).

    Returns:
        numpy.ndarray: Immagine con la linea più lunga disegnata.
    """

    # Crea una copia dell'immagine per disegnare la linea
    image_with_line = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)

    # Trova i contorni nell'immagine
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    longest_line = None  # La linea più lunga trovata
    max_length = 0

    for contour in contours:
        # Approssima il contorno con segmenti di linea
        approx = cv2.approxPolyDP(contour, epsilon=1, closed=False)

        for i in range(len(approx) - 1):
            # Calcola la lunghezza del segmento di linea
            x1, y1 = approx[i][0]
            x2, y2 = approx[i + 1][0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            lengthX = abs(x2-x1)
            lengthY = abs(y2-y1)
            
            #print(f"Lenght: {length}, LenghtX: {lengthX}")

            if lengthX > lengthY*2: #linea orizzontale
                if lengthX > max_length:
                    max_length = lengthX
                    #print(f"max_LenghtX: {max_length}")
                    longest_line = ((x1, y1), (x2, y2))
                    #cv2.line(image_with_line, longest_line[0], longest_line[1], (0, 0, 255), 2)  # Linea rossa di debug

    # Disegna la linea più lunga trovata
    if longest_line is not None:
        #print(f"Scelto: LenghtX: {max_length}")
        cv2.line(image_with_line, longest_line[0], longest_line[1], (0, 0, 255), 2)  # Linea rossa
        ((x1, y1), (x2, y2)) = longest_line
        angle_radians = np.arctan2(y2 - y1, x2 - x1)


    #print(f"Angolo pre-aggiustamento: {angle_radians}")
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
    # Crea una copia dell'immagine per disegnare la linea
    image_with_line = image.copy()

    # Ottieni i contorni validi
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 100]

    if not large_contours:
        return image_with_line  # Nessun contorno rilevante trovato

    # Trova i due punti più bassi
    lowerY, xPoint = -1, -1
    lowerY2, xPoint2 = -1, -1
    min_distance = 200

    for contour in large_contours:
        for point in contour:
            x, y = point[0]
            if y > lowerY or (y == lowerY and abs(x - xPoint) > min_distance):
                # Aggiorna il secondo punto più basso
                lowerY2, xPoint2 = lowerY, xPoint
                # Aggiorna il punto più basso
                lowerY, xPoint = y, x
            elif y > lowerY2 and abs(x - xPoint) > min_distance:
                # Aggiorna solo il secondo punto più basso
                lowerY2, xPoint2 = y, x

    if lowerY == -1 or lowerY2 == -1:
        return image_with_line  # Nessun punto trovato

    # Modifica l'altezza per l'angolo
    if abs(angle_radians) >= 0.5:
        lowerY -= 20

    start_x, start_y = xPoint, lowerY
    dx = xPoint2 - start_x
    dy = lowerY2 - start_y
    angle_radians = math.atan2(dy, dx)

    # Lunghezza della prolunga
    extension_length = 500  # Cambia questa lunghezza in base alle tue esigenze

    # Calcola i nuovi estremi della retta prolungata
    extended_start_x = int(start_x - extension_length * math.cos(angle_radians))
    extended_start_y = int(start_y - extension_length * math.sin(angle_radians))
    extended_end_x = int(xPoint2 + extension_length * math.cos(angle_radians))
    extended_end_y = int(lowerY2 + extension_length * math.sin(angle_radians))

    # Disegna la retta prolungata
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
    # Crea una copia dell'immagine per non modificare l'originale
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Converti l'immagine in un'immagine a colori
    found_polygons = []  # Per memorizzare i vertici dei poligoni trovati

    # Trova i contorni nell'immagine binaria (invertiamo l'immagine per cercare il nero)
    contours, _ = cv2.findContours(255 - image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ottieni le dimensioni dell'immagine
    height, width = image.shape
    center = None
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
                    #for corner in corners:
                        #cv2.circle(colored_image, corner, radius=10, color=(0, 0, 255), thickness=-1)  # Cerchio rosso
                        
                    center = find_center_of_polygon(corners)
                    center = (int(center[0]), int(center[1]))
                    #cv2.circle(colored_image, center, radius=10, color=(0, 255, 255), thickness=-1)

                    # Ritorna l'immagine, il booleano e il centro trovato
                    center = (int(center[0]) - (width/2), int(center[1]))
    #se vanno specchiate le coordinate
    # center = (-int(center[0]), int(center[1]))
    return colored_image, len(found_polygons) > 0, center

def find_center_of_polygon(corners):
    # Assumiamo che `corners` sia una lista di tuple con 4 punti, ad esempio [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    # Calcola la media delle coordinate x e y
    center_x = sum([corner[0] for corner in corners]) / len(corners)
    center_y = sum([corner[1] for corner in corners]) / len(corners)
    
    return (center_x, center_y)

def process_image(imageURL, i=0):
    # Assicurati che la cartella 'output' esista
    #output_dir = "./Federico/project/output"
    #os.makedirs(output_dir, exist_ok=True)  # Crea la cartella se non esiste

    # Preprocessing dell'immagine
    image_opened = preprocess_image(imageURL)  # Pre-processing dell'immagine
    angle_radians = (find_and_draw_longest_white_line(image_opened))
    #print(f"Angolo trovato: {angle_radians}")
    #image_with_lines = draw_line_at_angle(image_opened, angle_radians)  # Rilevamento delle linee
    image_with_lines = draw_line_at_angle_to_two_points(image_opened, angle_radians)
    #image_with_lines = draw_line_at_angle(image_with_lines, -angle_radians)
    image_parking_found, parking_exist, center = color_enclosed_black_areas(image_with_lines)

    #if parking_exist:
    #    print(f"Centro rilevato per il parcheggio: {center}")

    # Mostra l'immagine con la linea
    #cv2.imshow("Image with Horizontal Lines", image_parking_found)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Salva l'immagine risultante nella cartella 'output'
    #unique_filename = f"./output/image_" + str(i) + ".png"    
    #image_parking_found.save_to_disk(f'output/{unique_filename.frame}.png')
    #cv2.imwrite(unique_filename, image_parking_found)

    return parking_exist, center
