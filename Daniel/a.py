import carla, time, pygame, math, random, cv2
import numpy as np

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
client.load_world('Town5')
world = client.get_world()
spectator = world.get_spectator()