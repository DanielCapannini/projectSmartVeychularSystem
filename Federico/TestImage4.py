import cv2
import numpy as np
import random
from typing import List, Tuple, Optional


def extend_line(line, angle, length, direction="both"):
    x1, y1, x2, y2 = line

    # Calcolare la direzione della linea
    dx = x2 - x1
    dy = y2 - y1
    line_length = np.sqrt(dx**2 + dy**2)
    unit_dx = dx / line_length
    unit_dy = dy / line_length

    # Estendere la linea in entrambe le direzioni o in una direzione
    if direction == "both":
        x1 = int(x1 - unit_dx * length)
        y1 = int(y1 - unit_dy * length)
        x2 = int(x2 + unit_dx * length)
        y2 = int(y2 + unit_dy * length)
    elif direction == "start":
        x1 = int(x1 - unit_dx * length)
        y1 = int(y1 - unit_dy * length)
    elif direction == "end":
        x2 = int(x2 + unit_dx * length)
        y2 = int(y2 + unit_dy * length)

    return (x1, y1, x2, y2)


# Funzione per migliorare l'immagine (equalizzazione dell'istogramma)
def preprocess_image(imageURL):

    image = cv2.imread(imageURL)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #x_start, y_start = 0, 100  # Top-left corner
    #x_end, y_end = 1000, 900      # Bottom-right corner
    #gray_image = gray_image[y_start:y_end, x_start:x_end]

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
    
    # Visualizza l'immagine con le aree bianche isolate
    #cv2.imshow("Thresholded Image", image_opened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image_opened

def perpendicular_line_with_angle(x1, y1, x2, y2, length=50):
    """
    Calcola la perpendicolare alla retta definita dai punti (x1, y1) e (x2, y2), passando per il punto (x1, y1).
    Usa il seno e il coseno per ottenere la perpendicolare.
    
    :param x1: Coordinata x del punto 1 della retta
    :param y1: Coordinata y del punto 1 della retta
    :param x2: Coordinata x del punto 2 della retta
    :param y2: Coordinata y del punto 2 della retta
    :param length: Lunghezza della perpendicolare da disegnare (default: 50)
    :return: La perpendicolare come tupla (x1, y1), (x1_end, y1_end), (x2_end, y2_end)
    """
    # Calcola la direzione della retta
    dx = x2 - x1
    dy = y2 - y1

    # Calcola l'angolo della retta rispetto all'asse x
    theta = np.arctan2(dy, dx)

    # Calcola il vettore perpendicolare usando seno e coseno
    # Ruotiamo il vettore di 90 gradi (coseno, seno)
    perp_dx = -np.sin(theta)
    perp_dy = np.cos(theta)

    # Calcoliamo i punti finali della perpendicolare a partire da (x1, y1)
    x1_end = x1 + perp_dx * length
    y1_end = y1 + perp_dy * length

    return (x1, y1), (x1_end, y1_end)


def calculate_angle(line1: np.ndarray, line2: np.ndarray) -> float:
    """
    Calcola l'angolo in gradi tra due segmenti di linea.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Vettori direzione delle linee
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])

    # Calcola l'angolo tra i due vettori
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Limita il valore di cos_theta per evitare errori numerici
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def calculate_bisector(line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
    """
    Calcola la direzione della bisettrice dell'angolo tra due segmenti di linea.
    Restituisce un vettore unitario che rappresenta la direzione della bisettrice.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Vettori direzione delle linee
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])

    # Somma normalizzata dei vettori direzione
    bisector = v1 + v2
    bisector = bisector / np.linalg.norm(bisector)
    return bisector

def line_intersection(p1, p2, p3, p4):
    """
    Calcola il punto di intersezione di due linee definite dai punti p1-p2 e p3-p4.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if denom == 0:
        return None  # Le linee sono parallele, nessuna intersezione
    
    # Calcola le coordinate di intersezione
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return (int(x), int(y))

def is_point_in_array(point, points_array, tolerance=1):
    """
    Verifica se il punto (px, py) è presente nell'array dei punti esistenti.
    La funzione ora considera i punti vicini se sono entro una certa tolleranza (default 1).

    :param point: Il punto da verificare (px, py).
    :param points_array: L'array dei punti esistenti, ogni elemento è una tupla ((px, py), ...).
    :param tolerance: La tolleranza per considerare i punti "uguali".
    :return: True se il punto è presente nell'array, altrimenti False.
    """
    for p in points_array:
        # Estraiamo solo (px, py) da ogni elemento nell'array
        px, py = p[0]  # p[0] è il (px, py) del punto
        if np.isclose(px, point[0], atol=tolerance) and np.isclose(py, point[1], atol=tolerance):
            print(f"Punto {point} vicino a {p[0]}, quindi considerato uguale.")  # Debug
            return True
    return False

def find_intersections_between_arrays(array1, array2, points_array):
    """
    Trova le intersezioni tra le bisettrici in due array, aggiungendo solo quelle che non sono
    già presenti nell'array di punti.

    :param array1: Primo array di bisettrici, ciascuna definita da (x1, y1, x2, y2).
    :param array2: Secondo array di bisettrici, ciascuna definita da (x1, y1, x2, y2).
    :param points_array: Array di punti da escludere dalle intersezioni, ognuno definito da 
                          ((px, py), (bisector_end_x, bisector_end_y), angle).
    :return: Lista di intersezioni come tuple (px, py) che non sono nell'array points_array.
    """
    intersections = []

    for line1 in array1:
        for line2 in array2:
            # Trova l'intersezione tra line1 e line2
            intersection = find_line_intersection(line1, line2)
            if intersection is not None:
                # Se l'intersezione è una tupla di due valori (px, py), lo aggiungiamo
                if isinstance(intersection, tuple) and len(intersection) == 2:
                    # Verifica che l'intersezione non sia già nei punti esistenti
                    if not is_point_in_array(intersection, points_array):
                        intersections.append(intersection)

    return intersections

def find_line_intersection(line1, line2):
    """
    Calcola il punto di intersezione tra due rette estese indefinitamente, se esiste.

    :param line1: Prima linea definita da (x1, y1, x2, y2).
    :param line2: Seconda linea definita da (x1, y1, x2, y2).
    :return: Punto di intersezione (px, py) se esiste, altrimenti None.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    # Calcola il determinante
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Le rette sono parallele o coincidenti
    
    # Calcola le coordinate dell'intersezione
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    
    return int(px), int(py)  # Restituisci il punto di intersezione

def direction_to_line(direction_vector, origin, length=100):
    """
    Converte un vettore direzionale in una linea definita da due punti.
    
    :param direction_vector: Il vettore direzionale (come np.array).
    :param origin: Il punto di origine della retta (px, py).
    :param length: La lunghezza della retta (opzionale, default è 100).
    :return: Due punti (x1, y1), (x2, y2) che rappresentano la retta.
    """
    # Normalizza il vettore direzionale
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Calcola i punti finali della retta estendendo il vettore dalla posizione di origine
    x1, y1 = origin
    x2 = x1 + direction_vector[0] * length
    y2 = y1 + direction_vector[1] * length
    
    # Ritorna i due punti che definiscono la retta
    return (x1, y1, x2, y2)

def euclidean_distance(p1, p2):
    """
    Calcola la distanza Euclidea tra due punti p1 e p2.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def detect_lines(image):
    """
    Rileva linee nell'immagine, calcola intersezioni, angoli e bisettrici.
    """
    # Crea una copia dell'immagine in formato RGB (BGR -> RGB per OpenCV)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Applicare l'edge detection
    edges = cv2.Canny(image, 50, 150)

    # Applicare la trasformazione Hough per rilevare le linee
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=10)

    # Controlla se sono state trovate linee
    if lines is None:
        print("Nessuna linea trovata!")
        return

    # Raccogli tutte le linee
    all_lines = [(x1, y1, x2, y2) for x1, y1, x2, y2 in lines[:, 0]]

    # Estendi le linee
    extended_lines = [extend_line(line, 0, 100, direction="both") for line in all_lines]

    # Trova tutte le intersezioni tra i segmenti
    intersections = get_all_intersections(np.array(extended_lines))

    # Disegna le linee estese (su image_rgb)
    for x1, y1, x2, y2 in extended_lines:
        cv2.line(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Linea verde

    bisectors = []
    bisectors_specular = []
    # Disegna le intersezioni e le bisettrici
    for (px, py), (bisector_end_x, bisector_end_y), angle in intersections:
        
        add_bisector = True
        for (bx, by), (ex, ey) in bisectors:
            distance = euclidean_distance((bisector_end_x, bisector_end_y), (ex, ey))
            threshold_distance = 20
            if distance < threshold_distance:
                add_bisector = False
                break
        
        # Se la bisettrice è abbastanza lontana, la aggiungi
        if add_bisector:
            #trova la bisettrice speculare
            (dx, dy), (bisector_end_x_specular, bisector_end_y_specular) = perpendicular_line_with_angle(px, py, bisector_end_x, bisector_end_y)
            #aggiunge le bisettrici alle liste. Una per le bisettrici e le tangenti delle bisettrici
            bisectors.append(((px, py), (bisector_end_x, bisector_end_y)))
            bisectors_specular.append(((dx, dy), (bisector_end_x_specular, bisector_end_y_specular)))
            px, py = int(px), int(py)
            bisector_end_x, bisector_end_y = int(bisector_end_x), int(bisector_end_y)
            dx, dy = int(dx), int(dy)
            bisector_end_x_specular, bisector_end_y_specular = int(bisector_end_x_specular), int(bisector_end_y_specular)
            
            # Disegna il punto di intersezione
            cv2.circle(image_rgb, (int(px), int(py)), radius=5, color=(0, 0, 255), thickness=-1) # Disegna l'intersezione in rosso
            cv2.line(image_rgb, (px, py), (bisector_end_x, bisector_end_y), (255, 255, 0), 2)  # Bisettrice blu
            cv2.line(image_rgb, (dx, dy), (bisector_end_x_specular, bisector_end_y_specular ), (255, 255, 255), 2) #bisettrice speculare bianca
            # Stampa l'angolo
            #print(f"Angolo tra i segmenti: {angle:.2f}°")

    #bisectors_intersections = find_bisectors_intersections(bisectors, 0, 0)
    bisectors_intersections = find_intersections_between_arrays(bisectors, bisectors_specular, intersections)
    print("Intersezioni trovate:", bisectors_intersections)

    points = []
    for (x, y) in bisectors_intersections:
        # Verifica se il punto è all'interno dei limiti dell'immagine
        if 0 <= x < image_rgb.shape[1] and 0 <= y < image_rgb.shape[0]:
            points.append((x, y))
            # Disegna il cerchio in corrispondenza del punto di intersezione
            cv2.circle(image_rgb, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
        #else:
            #print(f"Warning: Punto fuori dai limiti: ({x}, {y})")
    
    find_and_color_closest_lines(image_rgb, points, extended_lines)
    # Mostra l'immagine finale con le linee, le intersezioni e le bisettrici
    cv2.imshow("Detected Lines, Intersections, and Bisectors", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_and_color_closest_lines(image, points, lines):
    """
    Trova e colora la linea sopra, a destra e a sinistra più vicina per ogni punto.
    :param image: Immagine su cui disegnare.
    :param points: Lista di punti, ogni punto è una tupla (px, py).
    :param lines: Lista di linee, ogni linea è una tupla ((x1, y1), (x2, y2)).
    """
    if len(image.shape) == 2:  # Se l'immagine è in scala di grigi
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image.copy()

    for px, py in points:
        closest_above = None
        closest_right = None
        closest_left = None
        min_distance_above = float('inf')
        min_distance_right = float('inf')
        min_distance_left = float('inf')

        for (x1, y1, x2, y2) in lines:
            # Verifica se il segmento interseca una linea tracciata sopra, a destra o a sinistra del punto
            if y1 <= py and y2 <= py:  # Segmento sopra
                dist = np.abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if dist < min_distance_above:
                    min_distance_above = dist
                    closest_above = ((x1, y1), (x2, y2))

            if x1 >= px and x2 >= px:  # Segmento a destra
                dist = np.abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if dist < min_distance_right:
                    min_distance_right = dist
                    closest_right = ((x1, y1), (x2, y2))

            if x1 <= px and x2 <= px:  # Segmento a sinistra
                dist = np.abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if dist < min_distance_left:
                    min_distance_left = dist
                    closest_left = ((x1, y1), (x2, y2))

        # Colora le linee più vicine con colori casuali
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if closest_above:
            cv2.line(image_rgb, closest_above[0], closest_above[1], random_color, 2)
        if closest_right:
            cv2.line(image_rgb, closest_right[0], closest_right[1], random_color, 2)
        if closest_left:
            cv2.line(image_rgb, closest_left[0], closest_left[1], random_color, 2)

        cv2.imshow("Detected Lines, Intersections, and Bisectors", image_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image_rgb



def remove_overlapping_lines(array1, array2, threshold=5):
    """
    Rimuove le linee nel secondo array che sono sovrapposte a quelle nel primo array.
    
    :param array1: Primo array di linee, ogni linea è definita da (x1, y1, x2, y2).
    :param array2: Secondo array di linee, ogni linea è definita da (x1, y1, x2, y2).
    :param threshold: Distanza massima per considerare due linee sovrapposte.
    :return: Un nuovo array2 senza linee sovrapposte.
    """
    def is_close(line1, line2, threshold):
        """
        Verifica se due linee sono sovrapposte entro un certo threshold.
        """
        # Calcolo della distanza media tra i punti di una linea e quelli dell'altra
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Controllo distanza tra i punti corrispondenti delle due linee
        dist1 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
        dist2 = np.sqrt((x2 - x4)**2 + (y2 - y4)**2)
        dist3 = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
        dist4 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)
        
        return min(dist1, dist2, dist3, dist4) < threshold

    filtered_array2 = []
    for line2 in array2:
        is_overlapping = False
        for line1 in array1:
            if is_close(line1, line2, threshold):
                is_overlapping = True
                break
        if not is_overlapping:
            filtered_array2.append(line2)
    
    return np.array(filtered_array2)


def find_segment_intersection(line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
    """
    Trova l'intersezione tra due segmenti di linea, se esiste.
    Restituisce il punto di intersezione (px, py) come array NumPy o un array vuoto se non c'è intersezione.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calcola i determinanti
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:
        return np.array([])  # Le linee sono parallele

    # Calcola le coordinate di intersezione
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    # Verifica se l'intersezione è all'interno dei segmenti
    if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4):
        return np.array([px, py])
    return np.array([])  # L'intersezione non è all'interno dei segmenti

def are_lines_overlapping(line1, line2, threshold):
    (x1, y1, x2, y2) = line1
    (x3, y3, x4, y4) = line2

    # Calcola distanze tra i punti corrispondenti
    dist1 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
    dist2 = np.sqrt((x2 - x4)**2 + (y2 - y4)**2)
    dist3 = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
    dist4 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)

    # Ritorna True se c'è una sovrapposizione entro il threshold
    return min(dist1, dist2, dist3, dist4) < threshold

def get_all_intersections(lines: np.ndarray) -> np.ndarray:
    """
    Trova tutte le intersezioni tra i segmenti di linee in un array e calcola le bisettrici e gli angoli.
    :param lines: Un array NumPy di forma (n, 4), dove n è il numero di linee.
    :return: Un array NumPy di tuple (px, py), (bisector_end_x, bisector_end_y), angle.
    """
    intersections_data = []

    first = True
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = find_segment_intersection(lines[i], lines[j])
            if intersection.size > 0:  # Se c'è una intersezione
                px, py = intersection


                angle = calculate_angle(lines[i], lines[j])
                    

                if angle > 10:  # Solo se l'angolo è maggiore di 10°
                    print(f"Angolo tra i segmenti: {angle:.2f}°")

                    # Calcola la bisettrice
                    if first:
                        bisector = calculate_bisector(lines[i], lines[j])
                        first = False

                    bisector_lines = direction_to_line(bisector, intersection)
                    
                    # Definisci la lunghezza della bisettrice
                    bisector_length = 50
                    bisector_end_x = px + bisector[0] * bisector_length
                    bisector_end_y = py + bisector[1] * bisector_length

                    is_overlapping = False
                    for line in lines:
                        if are_lines_overlapping(bisector_lines, line, 10):  # Se la bisettrice è sovrapposta
                            is_overlapping = True
                            break

                    if not is_overlapping:  # Solo se non è sovrapposta, aggiungiamo i dati
                        intersections_data.append(((px, py), (bisector_end_x, bisector_end_y), angle))

    return np.array(intersections_data)

def point_to_segment_distance(px, py, line):
    """
    Calcola la distanza minima tra un punto (px, py) e un segmento definito da 'line'.
    """
    x1, y1, x2, y2 = line
    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if line_length == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)  # La linea è un punto

    # Proiezione del punto sul segmento
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length ** 2)))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    # Distanza dal punto alla proiezione
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

def merge_parallel_lines(lines, angle_tolerance=5, distance_threshold=20):
    merged_lines = []
    used = set()

    for i, line1 in enumerate(lines):
        if i in used:
            continue

        x1, y1, x2, y2 = line1
        angle1 = np.arctan2(y2 - y1, x2 - x1)

        for j, line2 in enumerate(lines[i+1:], start=i+1):
            if j in used:
                continue

            x3, y3, x4, y4 = line2
            angle2 = np.arctan2(y4 - y3, x4 - x3)

            # Calcola la distanza minima tra le linee
            distance = min(
                np.linalg.norm([x1 - x3, y1 - y3]),
                np.linalg.norm([x1 - x4, y1 - y4]),
                np.linalg.norm([x2 - x3, y2 - y3]),
                np.linalg.norm([x2 - x4, y2 - y4]),
            )

            # Verifica se sono parallele e vicine
            if abs(np.degrees(angle1 - angle2)) < angle_tolerance and distance < distance_threshold:
                # Crea un rettangolo/romboidale unendo le linee
                merged_line = [
                    [x1, y1], [x2, y2],
                    [x4, y4], [x3, y3]
                ]
                merged_lines.append(np.array(merged_line, dtype=np.int32))
                used.add(i)
                used.add(j)
                break

        if i not in used:
            merged_lines.append(np.array([[x1, y1], [x2, y2], [x2, y2], [x1, y1]], dtype=np.int32))

    return merged_lines

def draw_center_circle(image, approx):
    # Calcola i momenti del contorno per trovare il centro del rettangolo
    M = cv2.moments(approx)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Disegna il cerchio giallo (BGR: [0, 255, 255]) al centro del rettangolo
        cv2.circle(image, (cx, cy), 10, (0, 255, 255), -1)  # Cerchio giallo con raggio 10


# Funzione principale per elaborare l'immagine
def process_image(imageURL):
    image_opened = preprocess_image(imageURL)  # Pre-processing dell'immagine
    detect_lines(image_opened)       # Rilevamento delle linee

# Esegui la funzione per le immagini
imageURL = "./Federico/img/parcheggio.jpg"
imageURLAlto = "./Federico/img/parcheggioAlto.jpg"
imageURL2 = "./Federico/img/parcheggio3.jpg"
ImageURLNew = "./RL/guida/output3/4365.png"
ImageURLNew2 = "./RL/guida/output3/4736.png"
ImageURLNew3 = "./RL/guida/output3/4750.png"

process_image(ImageURLNew)
process_image(ImageURLNew2)
process_image(ImageURLNew3)
