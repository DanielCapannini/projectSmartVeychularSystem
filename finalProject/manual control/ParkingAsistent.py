import carla, time, pygame, math, random, cv2
import numpy as np
from Utility import spawn_camera, speed_control, keep_lane, control_retro, find_point
from ParkingIdentifier import process_image

run = True
video_output=None 
center = None
firstTime = True
camera_retro = None
camera_rigth = None
radar = None
vehicle = None

min_ttc = float('inf')
def radar_callback(data: carla.RadarMeasurement):
    global min_ttc, min_distance
    min_ttc = float('inf')
    for detection, i in zip(data, range(len(data))):
        absolute_speed = abs(detection.velocity)
        if absolute_speed != 0:
            ttc = detection.depth / absolute_speed
            if ttc < min_ttc:
                min_ttc = ttc

def camera_rigth_callback(image):
    global run, video_output, center
    video_output = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    prediction, c=process_image(np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)))
    if prediction and run and c[0] > -100 and c[0] < 100:
        cv2.imwrite("prova.png", np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)))
        center = c
        print(center)
        run = False

def setParking(vehicleManual, world):
    global camera_retro, camera_rigth, radar, vehicle
    vehicle = vehicleManual
    radar_bp = world.get_blueprint_library().find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', '10')  # Horizontal field of view
    radar_bp.set_attribute('vertical_fov', '10')    # Vertical field of view
    radar_bp.set_attribute('range', '20')           # Maximum range
    radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
    radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
    camera_retro = spawn_camera(attach_to=vehicle, world=world, transform=carla.Transform(carla.Location(x=-1.6, z=1.9), carla.Rotation(yaw=180, pitch=-40)))
    camera_rigth = spawn_camera(attach_to=vehicle, world=world, transform=carla.Transform(carla.Location(x=-0.8, y=0.6, z=1.9), carla.Rotation(yaw=90, pitch=-40)))
    camera_rigth.listen(lambda image: camera_rigth_callback(image))


def parking(controlManual):
    target_speed_mps = 8 / 3.6
    control = controlManual
    control.steer = 0.0
    control.brake = 0.0
    target_distance = 3.2
    distance_travelled = 0.0
    n_listem_break=0
    ttc_threshold = 1.0
    collision = False
    time.sleep(1.5)
    while run:
        current_velocity = vehicle.get_velocity()
        current_speed_mps = current_velocity.length()
        control = keep_lane(video_output, control, current_speed_mps, target_speed_mps)
        vehicle.apply_control(vehicle, control)
        if min_ttc < ttc_threshold:
                control = carla.VehicleControl()
                control.brake = 1.0  
                vehicle.apply_control(vehicle, control)
                print("Emergency braking activated!")
                n_listem_break += 1
                if n_listem_break>10:
                    collision = True
                    break
    time.sleep(0.05)
    target_distance += -(center[0]/280)
    print(target_distance)
    pre_time = time.time()
    while distance_travelled < target_distance and not collision:
        current_velocity = vehicle.get_velocity()
        current_speed_mps = current_velocity.length()
        distance_travelled += current_speed_mps * (time.time() - pre_time)
        pre_time = time.time()
        control = keep_lane(video_output, control, current_speed_mps, target_speed_mps)
        vehicle.apply_control(control)
    control.brake = 1.0
    control.throttle = 0.0
    vehicle.apply_control(control)
    control.brake = 0.0
    control.reverse = True
    control.steer = 0.8
    distance_travelled = 0.0
    target_distance = 3.9
    pre_time = time.time()
    while distance_travelled < target_distance and not collision:
        current_velocity = vehicle.get_velocity()
        current_speed_mps = current_velocity.length()
        distance_travelled += current_speed_mps * (time.time() - pre_time)
        pre_time = time.time()
        control = speed_control(control, target_speed_mps, current_speed_mps)
        vehicle.apply_control(control)
    control.brake = 1.0
    control.throttle = 0.0
    vehicle.apply_control(control)
    camera_rigth.destroy()
    image1 = None
    def camera_callback2(image):
        global image1
        image1 =  np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    camera_retro.listen(lambda image: camera_callback2(image))
    time.sleep(0.15)
    while True:
        if control_retro(vehicle, find_point(image1), control, target_speed_mps=6/3.6):
            break
    control.brake = 1.0
    control.throttle = 0.0
    vehicle.apply_control(control)
    time.sleep(1)
    camera_retro.destroy()