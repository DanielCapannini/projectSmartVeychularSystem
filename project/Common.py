import cv2
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