import numpy as np
import cv2
import carla
import math
import time

template = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 13), np.uint8)
    opened_mask = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(opened_mask)
    for contour in contours:
        if cv2.contourArea(contour) >= 1500:
            cv2.drawContours(final_mask, [contour], -1, (255), thickness=cv2.FILLED)
    return final_mask

def preprocess_image2(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return mask_cleaned

def preprocess_image11(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    binary_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return mask_cleaned

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

def control_retro(vehicle, midpoint, control, target_speed_mps=10/3.6):
    if midpoint == None or midpoint[1] > 380:
        control.throttle = 0.0
        control.brake = 1.0
        vehicle.apply_control(control)
        time.sleep(0.1)
        return True
    else:
        control.steer = -((midpoint[0] / 400) - 1)
        current_velocity = vehicle.get_velocity()
        current_speed_mps = current_velocity.length()
        control = speed_control(control, target_speed_mps, current_speed_mps)
        vehicle.apply_control(control)
        time.sleep(0.1)
        return False

def speed_control(control, target_speed_mps, current_speed_mps):
    speed_error = target_speed_mps - current_speed_mps
    if speed_error > 0:
        control.throttle = min(1.0, 0.5 + speed_error * 0.5) 
        control.brake = 0
    elif speed_error < 0:
        control.throttle = 0.0
        control.brake = min(1.0, -speed_error * 0.5) 
    else:
        control.throttle = 0.0
        control.brake = 0.0
    return control

def find_point(img):
    img = preprocess_image(img)
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  
    cv2.line(output_img, (0, 450), (800, 350), (255, 255, 255), thickness=8) 
    img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    colored_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(255 - img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = img.shape
    areaM = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if x > 0 and y > 0 and (x + w) < width and (y + h) < height:
            if area > 1500: 
                print(area)
                areaM = area
                cv2.drawContours(colored_image, [contour], -1, (0,0,255), thickness=cv2.FILLED)
    mask = cv2.inRange(colored_image, np.array([0, 0, 255]), np.array([0, 0, 255]))
    mask = cv2.Canny(mask,100,200)
    midpoint, _, _ = find_highest_segment_midpoint_and_perpendicular(mask)
    if areaM > 10000:
        return midpoint
    elif areaM < 5000:
        return (400, midpoint[1])
    else:
        return midpoint

def find_highest_segment_midpoint_and_perpendicular(mask):
    """
    Trova il segmento più alto in un'immagine binaria utilizzando la trasformata di Hough, calcola
    il punto medio e determina una retta perpendicolare al segmento.

    Args:
        mask (numpy.ndarray): Immagine binaria (0 e 255).

    Returns:
        tuple: Coordinata (x, y) del punto medio e punti di inizio e fine della retta perpendicolare.
    """
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is None:
        return None, None, None
    highest_segment = None
    min_y = float('inf')
    for line in lines:
        x1, y1, x2, y2 = line[0]
        avg_y = (y1 + y2) / 2
        if avg_y < min_y:
            min_y = avg_y
            highest_segment = (x1, y1, x2, y2)
    if highest_segment is None:
        return None, None, None
    x1, y1, x2, y2 = highest_segment
    midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
    else:
        slope = float('inf')
    if slope != 0 and slope != float('inf'):
        perp_slope = -1 / slope
    else:
        perp_slope = 0 if slope == float('inf') else float('inf')
    length = 50 
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

def angle_calculation(image):
    imageP = preprocess_image(image)
    edges = cv2.Canny(imageP, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    min_angle = 75
    max_angle = 105
    min_angle_rad = np.deg2rad(min_angle)
    max_angle_rad = np.deg2rad(max_angle)
    theta_sum = 0
    line_count = 0
    if lines != None:
        return None
    for line in lines:
        rho, theta = line[0]
        if min_angle_rad <= theta <= max_angle_rad:
            theta_sum += theta
            line_count += 1
    if  line_count > 0:
        theta_mean = theta_sum / line_count
        theta_mean_deg = np.rad2deg(theta_mean) 
        if theta_mean_deg > 90:
            return theta_mean_deg+5
        if theta_mean_deg < 90:
            return theta_mean_deg-2
        return theta_mean_deg 
    return None

def keep_lane(image, control, current_speed_mps, target_speed_mps):
    angolo = angle_calculation(image)
    print(angolo)
    control.steer = 0.0
    if angolo != None:
        valore_scalato = (((angolo - 75) / (105 - 75)) * 0.8) - 0.4
        control.steer = valore_scalato
    control = speed_control(control, target_speed_mps, current_speed_mps)
    return control, angolo