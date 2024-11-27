import gym
from gym import spaces
import numpy as np
import carla
import random

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        # Connessione al simulatore Carla
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Impostazione dello spazio degli stati e delle azioni
        self.observation_space = spaces.Box(low=0, high=255, shape=(640, 480, 3), dtype=np.uint8)  # Immagine RGB di 640x480
        self.action_space = spaces.Discrete(4)  # 4 azioni: accelerare, frenare, sterzare sinistra, sterzare destra

        # Creazione di un veicolo (simulazione)
        self.vehicle = None

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
        reward = random.random()

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
        # Spawna un veicolo in un punto casuale nel mondo
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = self.world.spawn_actor(carla.client.get_world().get_blueprint_library().find('vehicle.tesla.model3'), spawn_point)
        return vehicle

    def _attach_camera(self, vehicle):
        # Attacca una telecamera al veicolo per ottenere l'immagine dello stato
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera = self.world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
        camera.listen(lambda image: self._process_image(image))
        return camera

    def _process_image(self, image):
        # Converte l'immagine in un formato adatto all'osservazione Gym (ad esempio un array NumPy)
        image.convert(carla.ColorConverter.Raw)
        self.image = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]

    def _get_observation(self):
        # Restituisce l'immagine (o altre informazioni) come stato
        return self.image
