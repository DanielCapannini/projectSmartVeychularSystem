import gym
from gym import spaces
import numpy as np
import carla
import random

from RL.Common import spawn_camera, preprocess_image, spawn_vehicle

camera_transforms = [
    (carla.Transform(carla.Location(x=1.5, z=2.4)), (600, 300)),  # Front camera
    (carla.Transform(carla.Location(x=-0.5, y=-0.9, z=2.4), carla.Rotation(yaw=-135)), (200, 400)),  # Left side camera
    (carla.Transform(carla.Location(x=-0.5, y=0.9, z=2.4), carla.Rotation(yaw=135)), (200, 400)),  # Right side camera
]

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.client.load_world('Town5')
        self.world = self.client.get_world()


        self.observation_space = spaces.Box(low=0, high=255, shape=(640, 480, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  # 4 azioni: accelerare, frenare, sterzare sinistra, sterzare destra

        # Creazione di un veicolo (simulazione)
        self.vehicle = None
        self.image = None

    def reset(self):
        # Resetta l'ambiente e il veicolo
        self.vehicle = self._spawn_vehicle()
        self.camera = self._attach_camera(self.vehicle)

        # Ritorna lo stato iniziale (in questo caso l'immagine dalla telecamera)
        state = self._get_observation()
        return state

    def step(self, action):
        # Esegui l'azione selezionata
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))  # Accelerazione
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))  # Frenata
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-1.0))  # Sterzare a sinistra
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=1.0))  # Sterzare a destra

        # Ottieni il nuovo stato (l'immagine della telecamera)
        state = self._get_observation()

        # Calcola la ricompensa (questa è una semplice ricompensa casuale)
        reward = random.random()#da implementare

        # Termina l'episodio (se il veicolo esce dalla strada, per esempio)
        done = random.random() > 0.95  # Termina con probabilità 5%

        return state, reward, done, {}

    def render(self, mode='human'):
        # Visualizza l'immagine dalla telecamera (opzionale, può essere utile per il debug)
        pass

    def close(self):
        # Chiudi la connessione con Carla
        if self.camera:
            self.camera.stop()
        if self.vehicle:
            self.vehicle.destroy()

    def _spawn_vehicle(self):
        return spawn_vehicle(self.world)

    def _attach_camera(self, vehicle):
        camera_list = np.array([])
        for i, transform in enumerate(camera_transforms):
            camera = spawn_camera(self.world, attach_to=vehicle, transform=transform[0], width=transform[1][0], height=transform[1][1])
            camera.listen(lambda image: self._process_image(image, i))
            camera_list = np.append(camera_list, camera)
        return camera_list

    def _process_image(self, image, index):
        self.image[index] = np.array(preprocess_image(image))

    def _get_observation(self):
        return self.image

    def collect_expert_data(env, esperto, num_episodi=10, file_path="expert_data.npz"):
        data = {"observations": [], "actions": []}
        for _ in range(num_episodi):
            obs = env.reset()
            done = False
            while not done:
                action = esperto(obs)  # L'azione definita dall'esperto
                data["observations"].append(obs)
                data["actions"].append(action)
                obs, _, done, _ = env.step(action)
        np.savez(file_path, **data)