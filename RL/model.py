import os
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from tensorflow import keras

from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Conv2D, Concatenate, Embedding, Reshape, Flatten, Activation, BatchNormalization
from keras.optimizers import SGD
from keras import regularizers

new_height = 300
new_width = 800


def create_model():
    # Image input
    image_input = Input(shape=(new_height, new_width, 3))
    # Integer input
    integer_input = Input(shape=(1,))
    # Preprocess the image input
    x = Conv2D(64, kernel_size=(6, 6), activation='relu',padding='same')(image_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(6, 6), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(6, 6), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(6, 6), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dense(8, activation='relu',activity_regularizer=regularizers.L2(1e-5))(x)
    x = Dropout(0.2)(x)
    x = Dense(4, activation='relu',activity_regularizer=regularizers.L2(1e-5))(x)
    x = Flatten()(x)
    # Concatenate image features with integer input
    concatenated_inputs = Concatenate()([x, integer_input])
    # Dense layers for prediction
    output = Dense(1, activation='linear')(concatenated_inputs)
    # Create the model
    model = Model(inputs=[image_input, integer_input], outputs=output)
    return model

# Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = np.where(thresholded == 255)
    sorted_pixels = np.sort(gray_image[white_pixels])
    threshold_value = sorted_pixels[int(0.85 * len(sorted_pixels))]
    _, custom_thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    mask = np.zeros_like(gray_image)
    mask[custom_thresholded == 255] = gray_image[custom_thresholded == 255]
    kernel = np.ones((3, 3), np.uint8)
    image_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    image_opened = cv2.morphologyEx(image_closed, cv2.MORPH_OPEN, (5,5))
    image_opened[image_opened > 0] = 255
    return image_opened