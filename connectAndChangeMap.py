import carla
import cv2
import numpy as np

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.load_world('Town05')