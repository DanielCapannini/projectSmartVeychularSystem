import carla
import cv2
import numpy as np

camera_transforms = [
    (carla.Transform(carla.Location(x=1.5, z=2.4)), (600, 300)),  # Front camera
    (carla.Transform(carla.Location(x=-0.5, y=-0.9, z=2.4), carla.Rotation(yaw=-135)), (200, 400)),  # Left side camera
    (carla.Transform(carla.Location(x=-0.5, y=0.9, z=2.4), carla.Rotation(yaw=135)), (200, 400)),  # Right side camera
    (carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=180)), (600, 300))  # Rear camera
]

client = carla.Client('localhost', 2000)
client.set_timeout(30.0)
world = client.get_world()
spectator = world.get_spectator()

# Funzione per il rilevamento delle linee bianche
def detect_white_lines(image):
    img_array = np.copy(image)
    img_bgr = img_array[:, :, :3]  # Ignora il canale alpha

    gray_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = np.where(thresholded == 255)
    sorted_pixels = np.sort(gray_image[white_pixels])
    threshold_value = sorted_pixels[int(0.90 * len(sorted_pixels))]
    _, custom_thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    mask = np.zeros_like(gray_image)
    mask[custom_thresholded == 255] = gray_image[custom_thresholded == 255]

    return mask

def create_camera_callback(index):
    def camera_callback(image):
        global video_outputs
        if index == 3:
            np_image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            video_outputs[index] = detect_white_lines(np_image)
        else:
            video_outputs[index] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    return camera_callback

def spawn_vehicle(vehicle_index=0, spawn_index=0, pattern='vehicle.*'):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter(pattern)[vehicle_index]
    spawn_point = world.get_map().get_spawn_points()[spawn_index]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    return vehicle

def spawn_camera(attach_to=None, transform=carla.Transform(carla.Location(x=1.2, z=1.2), carla.Rotation(pitch=-10)), width=800, height=600):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera = world.spawn_actor(camera_bp, transform, attach_to=attach_to)
    return camera

vehicle = spawn_vehicle()
vehicle.set_autopilot(True)

cameras = []
video_outputs = [np.zeros((600, 800, 4), dtype=np.uint8) for _ in range(4)]



for i, transform in enumerate(camera_transforms):
    camera = spawn_camera(attach_to=vehicle, transform=transform[0], width=transform[1][0], height=transform[1][1])
    camera.listen(create_camera_callback(i))
    cameras.append(camera)

# Configurazione delle finestre OpenCV
cv2.namedWindow('Front Camera', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Left Side Camera', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Right Side Camera', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Rear Camera', cv2.WINDOW_AUTOSIZE)

# Ciclo principale
running = True

try:
    while running:
        if cv2.waitKey(1) == ord('q'):
            running = False
            break
        cv2.imshow('Front Camera', video_outputs[0])
        cv2.imshow('Left Side Camera', video_outputs[1])
        cv2.imshow('Right Side Camera', video_outputs[2])
        cv2.imshow('Rear Camera', video_outputs[3])
finally:
    cv2.destroyAllWindows()
    for camera in cameras:
        camera.destroy()
    vehicle.destroy()