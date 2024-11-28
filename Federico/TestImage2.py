
import cv2
import numpy as np

# Funzione per migliorare l'immagine (equalizzazione dell'istogramma)
def preprocess_image(imageURL):
    image = cv2.imread(imageURL)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalizzazione dell'istogramma per migliorare il contrasto
    gray_image = cv2.equalizeHist(gray_image)

    # Threshold automatico per estrarre il bianco
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Estrazione dei bianchi con soglia personalizzata
    white_pixels = np.where(thresholded == 255)
    sorted_pixels = np.sort(gray_image[white_pixels])
    threshold_value = sorted_pixels[int(0.85 * len(sorted_pixels))]
    _, custom_thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Creazione della maschera per le aree bianche
    mask = np.zeros_like(gray_image)
    mask[custom_thresholded == 255] = 255  # Aree bianche
    mask[custom_thresholded != 255] = 0   # Aree non bianche (nero)

    # Operazioni morfologiche per migliorare l'immagine
    kernel = np.ones((5, 5), np.uint8)
    image_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    image_opened = cv2.morphologyEx(image_closed, cv2.MORPH_OPEN, (5,5))
    image_opened[image_opened > 0] = 255
    
    # Visualizza l'immagine con le aree bianche isolate
    #cv2.imshow("Thresholded Image", image_opened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image_opened

# Funzione per rilevare le linee nell'immagine usando la Trasformata di Hough
def detect_lines(image, original_image):
    # Converti l'immagine binaria in RGB (a 3 canali)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Converto l'immagine binaria in RGB per poter tracciare linee colorate

    # Applicazione di Canny per ottenere i bordi sull'immagine binaria
    edges = cv2.Canny(image, 50, 150)

    # Prova con una maggiore sensibilità per la trasformata di Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=10)

    if lines is None:
        print("Nessuna linea trovata!")
        return

    # Disegna tutte le linee sull'immagine RGB in rosso (BGR: [0, 0, 255])
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Linea rossa (BGR: [0, 0, 255])

    # Mostra l'immagine con le linee rosse tracciate sopra le aree bianche
    cv2.imshow("Detected Lines on White Areas", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Funzione principale per elaborare l'immagine
def process_image(imageURL):
    image_opened = preprocess_image(imageURL)  # Pre-processing dell'immagine
    detect_lines(image_opened, imageURL)       # Rilevamento delle linee

# Esegui la funzione per le immagini
imageURL = "./Federico/img/parcheggio.jpg"
imageURLAlto = "./Federico/img/parcheggioAlto.jpg"
imageURL2 = "./Federico/img/parcheggio3.jpg"

process_image(imageURL)
process_image(imageURL2)
process_image(imageURLAlto)