import numpy as np

def simple_expert(observation):
    front_camera = observation[0]  # Immagine dalla telecamera frontale
    left_camera = observation[1]   # Immagine dalla telecamera sinistra
    right_camera = observation[2]  # Immagine dalla telecamera destra
    
    if np.random.random() < 0.5:  # Per esempio, il 50% delle volte accelero
        return 0, 0, 0  # Accelerare
    else:
        return 0, 0, 1  # Frenare