import cv2
import numpy as np
import random
import os
import uuid
import math
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

def calcolo_angolo(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    min_angle = 75
    max_angle = 105
    min_angle_rad = np.deg2rad(min_angle)
    max_angle_rad = np.deg2rad(max_angle)
    theta_sum = 0
    line_count = 0
    for line in lines:
        rho, theta = line[0]
        if min_angle_rad <= theta <= max_angle_rad:
            theta_sum += theta
            line_count += 1
    theta_mean = theta_sum / line_count
    theta_mean_deg = np.rad2deg(theta_mean) 
    if theta_mean_deg > 100:
        return theta_mean_deg+5
    return abs(theta_mean_deg - 80)

def calculate_image_angle(image):

    # Rileva i bordi con Canny
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Trova le linee usando la trasformata di Hough
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Calcola gli angoli medi
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            print(f"angolo rilevato: {np.degrees(theta)}")
            if np.degrees(theta) > 75:
                angle = np.degrees(theta) - 90  # Converti l'angolo in gradi rispetto all'orizzontale
                angles.append(angle)

    # Calcola l'angolo medio
    if angles:
        average_angle = np.mean(angles)
    else:
        average_angle = 0

    return average_angle

def calculate_orientation(image):
    # Carica l'immagine in scala di grigi
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Applica threshold per ottenere un'immagine binaria
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Trova i contorni
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Trova l'inclinazione maggiore (approssimando un'ellisse)
    for contour in contours:
        if len(contour) >= 5:  # L'ellisse richiede almeno 5 punti
            ellipse = cv2.fitEllipse(contour)
            angle = ellipse[2]  # Angolo in gradi
            return angle

    return 0  # Nessun contorno trovato

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

    cv2.imshow("Image with Horizontal Lines", image_with_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return angle_radians

def draw_line_at_angle(image, angle_radians):
    """
    Trova il punto più basso di un pixel bianco in ogni colonna e disegna una linea inclinata
    di un angolo specifico rispetto all'orizzontale a partire da quel punto.

    Args:
        image (numpy.ndarray): Immagine binaria (0 e 255).
        angle (float): Angolo della linea in gradi (0 = orizzontale).

    Returns:
        numpy.ndarray: Immagine con la linea inclinata disegnata.
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

    # Scorri ogni colonna per trovare il pixel bianco più basso
    lowerY = 0
    xPoint = 0
    for x in range(width):
        for y in range(height - 1, -1, -1):
            if image[y, x] == 255:  # Trova il primo pixel bianco più basso
                # Verifica se il pixel fa parte di un contorno abbastanza grande
                for contour in large_contours:
                    if cv2.pointPolygonTest(contour, (x, y), False) >= 0:  # Il punto (x, y) è dentro il contorno
                        if lowerY < y:
                            lowerY = y
                            xPoint = x
                        break  # Fermati appena trovi il contorno valido

                break  # Fermati quando trovi il primo pixel bianco per colonna

    # Calcola i punti della linea inclinata
    L=width
    start_x = xPoint
    start_y = lowerY
    end_x = int(start_y + L * math.cos(angle_radians))
    end_y = int(lowerY - L * math.sin(angle_radians))
    end_x_neg = int(start_x - L * math.cos(angle_radians))
    end_y_neg = int(start_y + L * math.sin(angle_radians))
    
    # Disegna la linea inclinata
    cv2.line(image_with_line, (end_x_neg, end_y_neg), (end_x, end_y), (255, 255, 255), thickness=30)  # Linea bianca spessa

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
                    for corner in corners:
                        cv2.circle(colored_image, corner, radius=10, color=(0, 0, 255), thickness=-1)  # Cerchio rosso
                        
                    center = find_center_of_polygon(corners)
                    center = (int(center[0]), int(center[1]))
                    cv2.circle(colored_image, center, radius=10, color=(0, 255, 255), thickness=-1)

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
    print(f"Angolo trovato: {angle_radians}")
    image_with_lines = draw_line_at_angle(image_opened, angle_radians)  # Rilevamento delle linee
    #image_with_lines = draw_line_at_angle(image_with_lines, -angle_radians)
    image_parking_found, parking_exist, center = color_enclosed_black_areas(image_with_lines)

    if parking_exist:
        print(f"Centro rilevato per il parcheggio: {center}")

    # Mostra l'immagine con la linea
    cv2.imshow("Image with Horizontal Lines", image_parking_found)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Salva l'immagine risultante nella cartella 'output'
    #unique_filename = f"./output/image_" + str(i) + ".png"    
    #image_parking_found.save_to_disk(f'output/{unique_filename.frame}.png')
    #cv2.imwrite(unique_filename, image_parking_found)

    return parking_exist, center

ImageURLNew = "./RL/project/output/312363.png"
image = cv2.imread(ImageURLNew)
ImageURLNew2 = "./RL/project/output/312198.png"
image2 = cv2.imread(ImageURLNew2)
ImageURLNew3 = "./RL/project/output/312623.png"
image3 = cv2.imread(ImageURLNew3)
ImageURLNew4 = "./RL/guida/output3/4415.png"
image4 = cv2.imread(ImageURLNew4)

process_image(image)
process_image(image2)
process_image(image3)
process_image(image4)