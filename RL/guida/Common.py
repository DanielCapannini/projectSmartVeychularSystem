import numpy as np
import cv2
import carla
import math
import time

template = cv2.imread('output2/template.png', cv2.IMREAD_GRAYSCALE)

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = np.where(thresholded == 255)
    sorted_pixels = np.sort(gray_image[white_pixels])
    threshold_value = sorted_pixels[int(0.9 * len(sorted_pixels))]
    _, custom_thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    mask = np.zeros_like(gray_image)
    mask[custom_thresholded == 255] = gray_image[custom_thresholded == 255]
    kernel = np.ones((3, 3), np.uint8)
    image_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    image_opened = cv2.morphologyEx(image_closed, cv2.MORPH_OPEN, (5,5))
    image_opened[image_opened > 0] = 255
    return image_opened

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

def spawn_camera_depth(world, attach_to=None, transform=carla.Transform(carla.Location(x=1.2, z=1.2), carla.Rotation(pitch=-10, yaw=180)), width=800, height=600):
    camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera = world.spawn_actor(camera_bp, transform, attach_to=attach_to)
    return camera

def normalize_detph_image(image):
    normalized_depth = (image[:, :, 0] + image[:, :, 1] * 256 + image[:, :, 2] * 256**2) / (256**3 - 1)
    depth_in_meters = 1000 * normalized_depth
    print(depth_in_meters)
    depth_grayscale = cv2.normalize(depth_in_meters, None, 0, 255, cv2.NORM_MINMAX)
    depth_grayscale = depth_grayscale.astype(np.uint8) 
    return depth_grayscale

def follow_curve(world, vehicle, curve_points, speed=30):
    throttle = 0.5
    brake = 0.0
    steering = 0.0
    for point in curve_points:
        dx = point[0] - vehicle.get_location().x
        dy = point[1] - vehicle.get_location().y
        angle_to_target = math.atan2(dy, dx)
        angle_difference = angle_to_target - vehicle.get_transform().rotation.yaw
        if angle_difference > math.pi:
            angle_difference -= 2 * math.pi
        elif angle_difference < -math.pi:
            angle_difference += 2 * math.pi
        steering = max(-1.0, min(1.0, angle_difference / math.pi))
        if abs(angle_difference) < 0.1:
            throttle = 0.7
        else:
            throttle = 0.3
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steering, brake=brake, reverse=True))
        world.tick()
        time.sleep(0.1)

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
    t = np.linspace(0, 1, num_points)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    spline_x = h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0]
    spline_y = h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]

    return np.array(list(zip(spline_x, spline_y)), dtype=np.int32)

def riconosci_parcheggio(image):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    template_height, template_width = template.shape
    top_left = max_loc
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
    return max_val, top_left, bottom_right