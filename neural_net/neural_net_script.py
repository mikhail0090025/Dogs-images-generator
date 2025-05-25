import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import io
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tempfile
import random
import matplotlib.pyplot as plt

def get_global_variables():
    global images_shape, images
    try:
        # Get images shape
        images_shape_response = requests.get("http://dataset_manager:8000/images_shape")
        images_shape_response.raise_for_status()
        images_shape = images_shape_response.content
        print(f"Images shape: {images_shape}")

        # Get images
        learning_dataset_response = requests.get("http://dataset_manager:8000/get_images")
        learning_dataset_response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_file:
            temp_file.write(learning_dataset_response.content)
            temp_file_path = temp_file.name

        with open(temp_file_path, 'rb') as f:
            data = np.load(f)
            images = data['images']
            print(f"Loaded images: {images.shape}")

        import os
        os.remove(temp_file_path)

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

images_shape = None
images = None
get_global_variables()

def get_models():
    generator_net = keras.models.Sequential([
        keras.layers.Input(shape=(128,)),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        keras.layers.Dense(25*25*128, use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Reshape((25,25,128)),
        tf.keras.layers.Conv2DTranspose(128, 3, strides=(1, 1), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(32, 3, strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        
        tf.keras.layers.Conv2DTranspose(3, 3, strides=(1, 1), padding='same', use_bias=False, activation='tanh')
    ])
    descriminator_net = keras.models.Sequential([
        keras.layers.Input(shape=(100,100,3)),

        tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding='same'),
        keras.layers.Dropout(0.3),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(64, 3, strides=(2, 2), padding='same'),
        keras.layers.Dropout(0.3),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(128, 3, strides=(2, 2), padding='same'),
        keras.layers.Dropout(0.3),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    generator_net.summary()
    descriminator_net.summary()

    return generator_net, descriminator_net

generator_net, descriminator_net = get_models()

def generate_image():
    # Создаём шум с правильной формой (1, 128)
    noise = np.array([random.random() for _ in range(128)], dtype=np.float32)
    noise = noise.reshape(1, 128)  # Добавляем ось батча

    # Генерируем изображение
    generated = generator_net.predict(noise, verbose=1)
    generated = generated[0]  # Убираем ось батча, форма теперь (100, 100, 3)

    # Нормализуем значения в [0, 1]
    if generated.min() < 0:  # Если значения в [-1, 1] (tanh)
        generated = (generated + 1) / 2.0
    generated = np.clip(generated, 0, 1)  # Ограничиваем значения

    return generated

generate_image()