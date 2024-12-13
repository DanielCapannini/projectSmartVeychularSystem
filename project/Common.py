import cv2, math
import numpy as np
import carla

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return mask_cleaned

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
    image_with_line = image.copy()
    height, width = image.shape
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = []
    min_contour_area = 500
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            large_contours.append(contour)
    lowerY = 0
    for x in range(width):
        for y in range(height-1, -1, -1):
            if image[y, x] == 255:
                for contour in large_contours:
                    if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                        if lowerY < y:
                            lowerY = y
                        break
                break 
    cv2.line(image_with_line, (0, lowerY), (width-1, lowerY), (255, 255, 255), thickness=5)
    return image_with_line

def color_enclosed_black_areas(image, color=(0, 255, 0), min_area=500):
    """
    Trova le aree nere chiuse delimitate da righe bianche e le colora con un altro colore.
    
    Args:
        image (numpy.ndarray): Immagine binaria (0 e 255).
        color (tuple): Colore in formato BGR (default verde).
        min_area (int): L'area minima di un contorno per essere colorato (default 500).
    
    Returns:
        numpy.ndarray: Immagine con le aree nere colorate.
    """
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(255 - image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if x > 0 and y > 0 and (x + w) < width and (y + h) < height:
            if area > min_area: 
                cv2.drawContours(colored_image, [contour], -1, color, thickness=cv2.FILLED)
    return colored_image

def process_image(image):
    image_opened = preprocess_image(image)
    image_with_lines = draw_horizontal_line(image_opened)
    image_parking_found = color_enclosed_black_areas(image_with_lines)
    return image_parking_found

def spawn_vehicle(world, vehicle_index=0, pattern='vehicle.*'):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter(pattern)[vehicle_index]
    spawn_point = carla.Transform(carla.Location(-1, -30, 2), carla.Rotation(yaw=-90))
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    return vehicle

def spawn_camera(world, attach_to=None, transform=carla.Transform(carla.Location(x=1.2, z=1.2), carla.Rotation(pitch=-30)), width=800, height=600):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera = world.spawn_actor(camera_bp, transform, attach_to=attach_to)
    return camera

def spline_cubica(p0, t0, p1, t1, num_points=100):
    """
    Calcola una spline cubica tra due punti con tangenti specificate.
    
    :param p0: Punto iniziale (x0, y0)
    :param t0: Tangente al punto iniziale (tx0, ty0)
    :param p1: Punto finale (x1, y1)
    :param t1: Tangente al punto finale (tx1, ty1)
    :param num_points: Numero di punti per disegnare la curva
    :return: Lista di punti della spline cubica
    """
    t0 = np.array(t0) - np.array(p0) 
    t1 = np.array(t1) - np.array(p1)
    t = np.linspace(0, 1, num_points)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    spline_x = h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0]
    spline_y = h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]

    return np.array(list(zip(spline_x, spline_y)), dtype=np.int32)

def find_highest_segment_midpoint_and_perpendicular(mask):
    """
    Trova il segmento più alto in un'immagine binaria utilizzando la trasformata di Hough, calcola
    il punto medio e determina una retta perpendicolare al segmento.

    Args:
        mask (numpy.ndarray): Immagine binaria (0 e 255).

    Returns:
        tuple: Coordinata (x, y) del punto medio e punti di inizio e fine della retta perpendicolare.
    """
    # Trova linee con la trasformata di Hough
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is None:
        return None, None, None  # Nessuna linea trovata

    # Trova il segmento più alto
    highest_segment = None
    min_y = float('inf')

    for line in lines:
        x1, y1, x2, y2 = line[0]
        avg_y = (y1 + y2) / 2
        if avg_y < min_y:
            min_y = avg_y
            highest_segment = (x1, y1, x2, y2)

    if highest_segment is None:
        return None, None, None  # Nessun segmento valido trovato

    x1, y1, x2, y2 = highest_segment

    # Calcola il punto medio del segmento
    midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Calcola la pendenza del segmento
    if x2 != x1:  # Evita divisioni per zero
        slope = (y2 - y1) / (x2 - x1)
    else:
        slope = float('inf')  # Segmento verticale

    # Calcola la pendenza della retta perpendicolare
    if slope != 0 and slope != float('inf'):
        perp_slope = -1 / slope
    else:
        perp_slope = 0 if slope == float('inf') else float('inf')

    # Calcola i punti di inizio e fine della retta perpendicolare
    length = 50  # Lunghezza della retta perpendicolare (metà sopra e metà sotto il punto medio)
    if perp_slope == float('inf'):
        perp_start = (midpoint[0], midpoint[1] - length)
        perp_end = (midpoint[0], midpoint[1] + length)
    elif perp_slope == 0:
        perp_start = (midpoint[0] - length, midpoint[1])
        perp_end = (midpoint[0] + length, midpoint[1])
    else:
        dx = int(length / math.sqrt(1 + perp_slope**2))
        dy = int(perp_slope * dx)
        perp_start = (midpoint[0] - dx, midpoint[1] - dy)
        perp_end = (midpoint[0] + dx, midpoint[1] + dy)

    return midpoint, perp_start, perp_end