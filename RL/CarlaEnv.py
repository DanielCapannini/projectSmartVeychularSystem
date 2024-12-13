import gym
from gym import spaces
import numpy as np
import carla
import random
import cv2

from Common import spawn_camera, preprocess_image, spawn_vehicle

camera_transforms = [
    (carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-25)), (600, 300)),  # Front camera
    (carla.Transform(carla.Location(x=-0.5, y=-0.9, z=2.4), carla.Rotation(yaw=-45, pitch=-25)), (600, 300)),  # Left side camera
    (carla.Transform(carla.Location(x=-0.5, y=0.9, z=2.4), carla.Rotation(yaw=45, pitch=-25)), (600, 300)),  # Right side camera

]

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.client.load_world('Town05')
        self.world = self.client.get_world()

        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 300, 600, 3), dtype=np.uint8)  # Shape is (3 cameras, 300x600 px, 3 color channels)
        self.action_space = spaces.Discrete(4)  # 4 actions: accelerate, brake, steer left, steer right

        self.vehicle = None
        self.camera = []
        self.image1 = np.zeros((50, 100))
        self.image2 = np.zeros((50, 100))
        self.image0 = np.zeros((50, 100))
        self.counter = 0

    def reset(self):
        # Reset the environment, spawn a new vehicle, and attach cameras
        self.vehicle = self._spawn_vehicle()
        self.camera = self._attach_camera(self.vehicle)

        # Return the initial state (images from the cameras)
        state = self._get_observation()
        self.counter = 0
        return state

    def step(self, throttle, steer, brake):
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        state = self._get_observation()
        reward = self.render()
        done = self.done()
        return state, reward, done, {}
    
    def reward(self):
        return 0
    
    
    def done(self):
        self.counter += 1
        if self.counter >= 3000:
            self.vehicle.destroy()
            return True
        else:
            return False

    def render(self, mode='human'):
        pass

    def close(self):
        if self.camera:
            for cam in self.camera:
                cam.stop()
        if self.vehicle:
            self.vehicle.destroy()

    def _spawn_vehicle(self):
        return spawn_vehicle(self.world)

    def _attach_camera(self, vehicle):
        camera_list = []
        for i, transform in enumerate(camera_transforms):
            camera = spawn_camera(self.world, attach_to=vehicle, transform=transform[0], width=transform[1][0], height=transform[1][1])
            camera.listen(lambda image, idx=i: self._process_image(image, idx))
            camera_list.append(camera)
        return camera_list

    def _process_image(self, image, index):
        image_output_left = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        processed_image = np.array(preprocess_image(image_output_left))
        resized_image = cv2.resize(processed_image, (100, 50))
        print(np.shape(self.image0))
        print(np.shape(self.image1))
        print(np.shape(self.image2))
        if index==0:
            self.image0 = resized_image
        if index==1:
            self.image1 = resized_image
        if index==2:
            self.image2 = resized_image

    def _get_observation(self):
        return np.stack((self.image0, self.image1, self.image2), axis=0)

    @staticmethod
    def collect_expert_data(env, expert, num_episodes=10, file_path="./expert_data.npz"):
        data = {"observations": [], "actions": []}
        #print("start epoch")
        for i in range(num_episodes):
            #print("start "+ str(i)+ " epoch")
            obs = env.reset()
            done = False
            while not done:
                throttle, steer, brake = expert(obs)
                data["observations"].append(obs)
                data["actions"].append((throttle, steer, brake))
                obs, _, done, _ = env.step(throttle, steer, brake)
            #print("end "+ str(i)+ " epoch")
        np.savez(file_path, **data)
