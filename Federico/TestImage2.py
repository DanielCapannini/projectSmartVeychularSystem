
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
    intersections = find_intersections(merged_lines)

    # Disegna i rettangoli/romboidi
    for merged_line in merged_lines:
        cv2.polylines(image_rgb, [np.array(merged_line, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    # Disegna i punti di intersezione
    for (px, py), (bisector_end_x, bisector_end_y), angle in intersections:
        # Disegna il punto di intersezione
        cv2.circle(image_rgb, (px, py), radius=5, color=(0, 255, 255), thickness=-1)  # Cerchio giallo
        # Disegna la bisettrice
        cv2.line(image_rgb, (px, py), (bisector_end_x, bisector_end_y), (0, 255, 0), 2)  # Bisettrice verde
        # Stampa l'angolo
        print(f"Angolo tra i segmenti: {angle:.2f}°")

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

def find_intersections(lines):
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
                        intersections.append((intersection, bisector_end, angle))

    return intersections

def draw_bisectors(image, intersections, lines):
    """
    Disegna le bisettrici dei segmenti che si intersecano in ciascun punto di intersezione.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for px, py in intersections:
        # Trova i segmenti che si intersecano in (px, py)
        intersecting_segments = []
        for line in lines:
            for seg in [
                (line[0][0], line[0][1], line[1][0], line[1][1]),
                (line[1][0], line[1][1], line[2][0], line[2][1]),
                (line[2][0], line[2][1], line[3][0], line[3][1]),
                (line[3][0], line[3][1], line[0][0], line[0][1]),
            ]:
                x1, y1, x2, y2 = seg
                # Controlla se (px, py) è sul segmento
                if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2):
                    intersecting_segments.append(seg)

        if len(intersecting_segments) < 2:
            continue  # Serve almeno due segmenti per calcolare la bisettrice

        # Calcola i vettori dei segmenti che si intersecano
        vectors = []
        for x1, y1, x2, y2 in intersecting_segments:
            if (x1, y1) == (px, py):
                vectors.append((x2 - x1, y2 - y1))
            elif (x2, y2) == (px, py):
                vectors.append((x1 - x2, y1 - y2))
            else:
                continue

        if len(vectors) < 2:
            continue  # Se non ci sono due vettori validi, salta

        # Calcola la direzione della bisettrice
        v1 = unit_vector(vectors[0])
        v2 = unit_vector(vectors[1])
        bisector = unit_vector((v1[0] + v2[0], v1[1] + v2[1]))

        # Disegna la bisettrice
        end_x = int(px + bisector[0] * 50)  # Estendi la bisettrice
        end_y = int(py + bisector[1] * 50)
        cv2.arrowedLine(image_rgb, (px, py), (end_x, end_y), color=(0, 255, 255), thickness=2, tipLength=0.3)

    # Mostra l'immagine con le bisettrici
    cv2.imshow("Detected Lines, Intersections, and Bisectors", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
process_image(imageURLAlto)
