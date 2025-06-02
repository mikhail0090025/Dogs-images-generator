import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import io
import tempfile
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Глобальные переменные
noise_size = 512
generator_net = None
discriminator_net = None
gan = None
images = None
images_shape = None
d_losses = []
g_losses = []
print("Variables are reset")

def get_global_variables():
    global images_shape, images
    try:
        images_shape_response = requests.get("http://dataset_manager:8000/images_shape")
        images_shape_response.raise_for_status()
        images_shape = images_shape_response.content
        print(f"Images shape: {images_shape}")

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

def get_models():
    global discriminator_net, generator_net
    generator_net = keras.models.Sequential([
        keras.layers.Input(shape=(noise_size,)),
        keras.layers.Dense(2*2*1024, use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((2,2,1024)),

        # Первый слой: 2x2 → 4x4
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        # Второй слой: 4x4 → 8x8
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        # Третий слой: 8x8 → 16x16
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        # Четвёртый слой: 16x16 → 32x32
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(192, 3, padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        # Пятый слой: 32x32 → 64x64
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        # Шестой слой: 64x64 → 128x128
        # tf.keras.layers.UpSampling2D((2, 2)),
        # tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False),
        # keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(),

        # Выходной слой: 128x128x3
        tf.keras.layers.Conv2D(3, 3, padding='same', use_bias=False, activation='sigmoid')
    ])
    discriminator_net = keras.models.Sequential([
        keras.layers.Input(shape=(64,64,3)),

        tf.keras.layers.Conv2D(256, 3, strides=(2, 2), padding='same'),
        #keras.layers.Dropout(0.2),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(512, 3, strides=(2, 2), padding='same'),
        #keras.layers.Dropout(0.2),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(1024, 3, strides=(2, 2), padding='same'),
        #keras.layers.Dropout(0.1),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(2048, 3, strides=(2, 2), padding='same'),
        #keras.layers.Dropout(0.1),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(2048, 3, strides=(1,1), padding='same'),
        #keras.layers.Dropout(0.1),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),

        keras.layers.Dense(2048, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),

        keras.layers.Dense(1, activation='sigmoid'),
    ])

    optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
    optimizer_d = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)

    generator_net.compile(optimizer=optimizer_g, loss='binary_crossentropy')
    discriminator_net.compile(optimizer=optimizer_d, loss='binary_crossentropy')

    generator_net.summary()
    discriminator_net.summary()

    return generator_net, discriminator_net

def generate_image():
    global generator_net
    if generator_net is None:
        raise ValueError("Generator network is not initialized. Call main() first.")
    noise = np.random.normal(0, 1, (1, noise_size))

    generated = generator_net.predict(noise, verbose=1)
    generated = generated[0]
    factor = max(generated.max(), abs(generated.min()))
    # generated = generated / factor
    print("Data min/max:", generated.min(), generated.max())

    if generated.min() < 0:
        generated = (generated + 1) / 2.0
    generated = np.clip(generated, 0, 1)

    return generated

def go_one_epoch(batch_size, generator_net, discriminator_net, gan):
    global images, d_losses, g_losses
    if images is None:
        raise ValueError("Images are not loaded. Call get_global_variables() first.")
    idx = np.random.randint(0, images.shape[0], batch_size)
    real_imgs = images[idx]

    noise = np.random.normal(0, 1, (batch_size, noise_size))
    fake_imgs = generator_net.predict(noise, verbose=1)

    print("Fake imgs min/max:", fake_imgs.min(), fake_imgs.max())

    generator_net.trainable = False
    d_loss_real = discriminator_net.train_on_batch(real_imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator_net.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
    generator_net.trainable = True
    d_loss = 0.5 * (d_loss_real + d_loss_fake)

    discriminator_net.trainable = False
    noise = np.random.normal(0, 1, (batch_size, noise_size))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    discriminator_net.trainable = True

    d_losses.append(d_loss)
    g_losses.append(g_loss)

    return d_loss, g_loss

def go_one_epoch_generator(batch_size, generator_net, discriminator_net, gan):
    global images, d_losses, g_losses
    if images is None:
        raise ValueError("Images are not loaded. Call get_global_variables() first.")

    noise = np.random.normal(0, 1, (batch_size, noise_size))
    fake_imgs = generator_net.predict(noise, verbose=1)

    print("Fake imgs min/max:", fake_imgs.min(), fake_imgs.max())

    noise = np.random.normal(0, 1, (batch_size, noise_size))
    discriminator_net.trainable = False
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    discriminator_net.trainable = True

    g_losses.append(g_loss)

    return g_loss

def go_one_epoch_discriminator(batch_size, generator_net, discriminator_net, gan):
    global images, d_losses, g_losses
    if images is None:
        raise ValueError("Images are not loaded. Call get_global_variables() first.")
    idx = np.random.randint(0, images.shape[0], batch_size)
    real_imgs = images[idx]

    noise = np.random.normal(0, 1, (batch_size, noise_size))
    fake_imgs = generator_net.predict(noise, verbose=1)

    print("Fake imgs min/max:", fake_imgs.min(), fake_imgs.max())

    generator_net.trainable = False
    d_loss_real = discriminator_net.train_on_batch(real_imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator_net.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
    generator_net.trainable = True
    d_loss = 0.5 * (d_loss_real + d_loss_fake)

    d_losses.append(d_loss)
    g_losses.append(g_losses[-1])

    return d_loss

def plot_losses():
    fig = go.Figure(
        data=[
            go.Scatter(x=list(range(len(g_losses))), y=g_losses, name='Generator loss', line=dict(color='green')),
            go.Scatter(x=list(range(len(d_losses))), y=d_losses, name='Discriminator loss', line=dict(color='orange')),
        ],
        layout={
            'title': {'text': 'Loss'},
            'yaxis': {'type': 'log', 'title': 'Loss (Log Scale)'},
            'xaxis': {'title': 'Epoch'}
        }
    )

    return fig

def go_epochs(epochs_count, batch_size):
    global generator_net, discriminator_net, gan
    if any(x is None for x in [generator_net, discriminator_net, gan]):
        raise ValueError("Models are not initialized. Call main() first.")
    for i in range(epochs_count):
        d_loss, g_loss = go_one_epoch(batch_size, generator_net, discriminator_net, gan)
        print(f"Epoch {i+1}/{epochs_count}: D Loss: {d_loss}, G Loss: {g_loss}")
    print(f"{epochs_count} epochs passed!")

def go_epochs_discriminator(epochs_count, batch_size):
    global generator_net, discriminator_net, gan
    if any(x is None for x in [generator_net, discriminator_net, gan]):
        raise ValueError("Models are not initialized. Call main() first.")
    for i in range(epochs_count):
        d_loss = go_one_epoch_discriminator(batch_size, generator_net, discriminator_net, gan)
        print(f"Epoch {i+1}/{epochs_count}: D Loss: {d_loss}")
    print(f"{epochs_count} epochs passed!")

def main():
    global gan, generator_net, discriminator_net
    get_global_variables()
    generator_net, discriminator_net = get_models()
    gan_input = tf.keras.layers.Input(shape=(noise_size,))
    gan_output = discriminator_net(generator_net(gan_input))
    gan = tf.keras.Model(gan_input, gan_output)
    print("Variables are defined")
    gan.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005), loss='binary_crossentropy')

if __name__ == '__main__':
    main()