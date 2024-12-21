
import cv2
import numpy as np
import math


def unit_vector(v):
    """
    Calcola il vettore unitario.
    """
    length = np.sqrt(v[0]**2 + v[1]**2)
    if length == 0:
        return (0, 0)
    return (v[0] / length, v[1] / length)

def is_point_within_image(x, y, image_width, image_height):
    """
    Verifica se il punto (x, y) è all'interno dei limiti dell'immagine.
    """
    return 0 <= x < image_width and 0 <= y < image_height

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

def detect_lines_and_draw_intersections(image, original_image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=10)

    if lines is None:
        print("Nessuna linea trovata!")
        return

    all_lines = [(x1, y1, x2, y2) for x1, y1, x2, y2 in lines[:, 0]]

    # Estensione delle linee
    extended_lines = [extend_line(line, 0, 50, direction="both") for line in all_lines]

    # Unisci linee parallele vicine
    merged_lines = merge_parallel_lines(extended_lines)

    # Trova i punti di intersezione
    intersections = find_intersections(merged_lines, width, height)

    # Disegna i rettangoli/romboidi
    for merged_line in merged_lines:
        cv2.polylines(image_rgb, [np.array(merged_line, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    bisectors = [] 
    #bisectors_intersection = []

    # Disegna i punti di intersezione
    for (px, py), (bisector_end_x, bisector_end_y), angle in intersections:
        bisectors.append(((px, py), (bisector_end_x, bisector_end_y)))
        # Disegna il punto di intersezione
        cv2.circle(image_rgb, (px, py), radius=5, color=(0, 255, 255), thickness=-1)  # Cerchio giallo
        # Disegna la bisettrice
        cv2.line(image_rgb, (px, py), (bisector_end_x, bisector_end_y), (0, 255, 0), 2)  # Bisettrice verde
        # Stampa l'angolo
        #print(f"Angolo tra i segmenti: {angle:.2f}°")

    bisectors_intersections = find_bisectors_intersections(bisectors, width, height)

    remove_nearby_points(intersections, bisectors_intersections)

    for (x, y) in bisectors_intersections:
        # Verifica se il punto è all'interno dei limiti dell'immagine
        if 0 <= x < image_rgb.shape[1] and 0 <= y < image_rgb.shape[0]:
            # Disegna il cerchio in corrispondenza del punto di intersezione
            cv2.circle(image_rgb, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
        #else:
            #print(f"Warning: Punto fuori dai limiti: ({x}, {y})")

    # Mostra l'immagine finale
    cv2.imshow("Detected Lines, Intersections, and Angles", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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

def find_intersections(lines, image_width, image_height):
    """
    Trova i punti di intersezione tra i lati di poligoni (rettangoli o romboidi).
    Calcola anche l'angolo tra i segmenti che si intersecano e disegna la bisettrice.
    """
    intersections = []

    def segment_intersection(seg1, seg2):
        """
        Calcola il punto di intersezione tra due segmenti (se esiste),
        calcola anche l'angolo tra di essi e disegna la bisettrice.
        """
        x1, y1 = seg1[0]
        x2, y2 = seg1[1]
        x3, y3 = seg2[0]
        x4, y4 = seg2[1]

        # Determinanti per verificare l'intersezione
        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if det == 0:
            return None, None, None  # Segmenti paralleli, nessuna intersezione

        # Coordinate dell'intersezione
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

        # Verifica che il punto si trovi all'interno dei segmenti
        if (
            min(x1, x2) <= px <= max(x1, x2)
            and min(y1, y2) <= py <= max(y1, y2)
            and min(x3, x4) <= px <= max(x3, x4)
            and min(y3, y4) <= py <= max(y3, y4)
        ):
            # Calcola i vettori direzionali dei segmenti
            v1 = (x2 - x1, y2 - y1)  # Vettore del primo segmento
            v2 = (x4 - x3, y4 - y3)  # Vettore del secondo segmento

            # Calcola il prodotto scalare
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]

            # Calcola la magnitudine dei vettori
            mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

            # Calcola l'angolo tra i vettori in radianti
            cos_angle = dot_product / (mag_v1 * mag_v2)
            angle_rad = math.acos(cos_angle)  # Angolo in radianti
            angle_deg = math.degrees(angle_rad)  # Angolo in gradi

            # Calcola la bisettrice: somma dei vettori direzionali normalizzati
            bisector_x = v1[0] / mag_v1 + v2[0] / mag_v2
            bisector_y = v1[1] / mag_v1 + v2[1] / mag_v2

            # Normalizzare la bisettrice
            mag_bisector = math.sqrt(bisector_x**2 + bisector_y**2)
            bisector_x /= mag_bisector
            bisector_y /= mag_bisector

            # Estendere la bisettrice a partire dall'intersezione
            bisector_length = 50  # Puoi cambiare la lunghezza della bisettrice
            bisector_end_x = int(px + bisector_x * bisector_length)
            bisector_end_y = int(py + bisector_y * bisector_length)

            return (int(px), int(py)), (bisector_end_x, bisector_end_y), angle_deg

        return None, None, None

    # Ciclo su tutte le coppie di segmenti in tutti i romboidi
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            # Itera su tutti i segmenti del primo e secondo romboide
            for k in range(0, 4, 2):  # Ogni romboide ha 4 segmenti
                seg1 = [lines[i][k], lines[i][k + 1]]
                for l in range(0, 4, 2):
                    seg2 = [lines[j][l], lines[j][l + 1]]
                    intersection, bisector_end, angle = segment_intersection(seg1, seg2)
                    if intersection:
                        # Controlla se il punto di intersezione è dentro l'immagine
                        px, py = intersection
                        if is_point_within_image(px, py, image_width, image_height):
                            intersections.append((intersection, bisector_end, angle))
                        else:
                            print(f"Punto fuori dai limiti dell'immagine: {intersection}")
    
    return intersections

def find_bisectors_intersections(bisectors, image_width, image_height, max_distance=0.5):
    intersections = []
    
    for i in range(len(bisectors)):
        for j in range(i + 1, len(bisectors)):
            # Prendi due bisettrici (ogni bisettrice è una coppia di punti)
            (x1, y1), (x2, y2) = bisectors[i]
            (x3, y3), (x4, y4) = bisectors[j]
            
            # Calcola il punto di intersezione tra le due linee (bisettrici)
            intersection_point = line_intersection((x1, y1), (x2, y2), (x3, y3), (x4, y4))
            
            if intersection_point:
                # Verifica se l'intersezione è entro la distanza massima dalle due linee
                (ix, iy) = intersection_point
                dist1 = distance_point_to_line(ix, iy, x1, y1, x2, y2)
                dist2 = distance_point_to_line(ix, iy, x3, y3, x4, y4)
                
                # Aggiungi l'intersezione solo se è entro la distanza massima dalle linee
                if dist1 <= max_distance and dist2 <= max_distance:
                    # Controlla se il punto di intersezione è dentro l'immagine
                    if is_point_within_image(ix, iy, image_width, image_height):
                        intersections.append(intersection_point)
                    else:
                        print(f"Punto di intersezione fuori dai limiti dell'immagine: {intersection_point}")
    
    return intersections

def distance_point_to_line(px, py, x1, y1, x2, y2):
    """
    Calcola la distanza tra un punto (px, py) e la linea definita dai punti (x1, y1) e (x2, y2).
    """
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denom = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    distance = num / denom
    print(f"Distanza tra ({px}, {py}) e la linea: {distance}")  # Debug
    
    return distance

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

def points_distance(p1, p2):
    """
    Calcola la distanza euclidea tra due punti p1 e p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # Stampa la distanza trovata per il debug
    #print(f"Distanza tra {p1} e {p2}: {distance}")
    
    return distance

def remove_nearby_points(array1, array2, range_distance = 40):
    """
    Controlla se i punti nel primo array (array1) sono vicini ai punti nel secondo array (array2),
    e rimuove i punti vicini da array2.
    
    :param array1: Primo array di punti e bisettrici, dove ogni elemento è del tipo
                   ((px, py), (bisector_end_x, bisector_end_y), angle_deg).
    :param array2: Secondo array di punti (array di tuple (x, y)).
    :param range_distance: Range di distanza entro il quale i punti sono considerati "vicini".
    
    :return: array2 modificato.
    """
    to_remove = []  # Lista di indici da rimuovere in array2
    
    # Scorri ogni elemento di array1
    for element1 in array1:
        (px, py), (bisector_end_x, bisector_end_y), _ = element1  # Estrai i punti (px, py) e (bisector_end_x, bisector_end_y)
        
        # Calcola la distanza tra (px, py) e (bisector_end_x, bisector_end_y)
        for i, point2 in enumerate(array2):
            (x2, y2) = point2
            
            # Verifica se il punto del secondo array è vicino a uno dei punti del primo array
            distance = points_distance((px, py), (x2, y2))
            
            # Se almeno uno dei punti del primo array è vicino al punto del secondo array, lo rimuoviamo
            if distance <= range_distance:
                del array2[i]
    
    """
    # Rimuovi i punti da array2, senza alterare l'indice durante l'iterazione
    for index in sorted(to_remove, reverse=True):
        try:
            del array2[index]
        except IndexError:
            print()
            #print(f"Errore: Tentativo di rimuovere l'indice {index} che non esiste.")
    """
    return array2

# Funzione principale per elaborare l'immagine
def process_image(imageURL):
    image_opened = preprocess_image(imageURL)  # Pre-processing dell'immagine
    detect_lines_and_draw_intersections(image_opened, imageURL)       # Rilevamento delle linee

# Esegui la funzione per le immagini
imageURL = "./Federico/img/parcheggio.jpg"
imageURLAlto = "./Federico/img/parcheggioAlto.jpg"
imageURL2 = "./Federico/img/parcheggio3.jpg"

process_image(imageURL)
process_image(imageURL2)
#process_image(imageURLAlto)
