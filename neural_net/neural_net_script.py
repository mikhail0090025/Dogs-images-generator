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
        keras.layers.Input(shape=(128,)),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        keras.layers.Dense(10*10*512, use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((10, 10, 512)),

        tf.keras.layers.Conv2DTranspose(512, 3, strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(512, 3, strides=(1, 1), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        
        tf.keras.layers.Conv2DTranspose(256, 3, strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(256, 3, strides=(1, 1), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        
        tf.keras.layers.Conv2DTranspose(128, 3, strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(128, 3, strides=(1, 1), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        
        tf.keras.layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(64, 3, strides=(1, 1), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        
        tf.keras.layers.Conv2DTranspose(3, 3, strides=(1, 1), padding='same', use_bias=False, activation='sigmoid')
    ])
    discriminator_net = keras.models.Sequential([
        keras.layers.Input(shape=(160, 160, 3)),

        tf.keras.layers.Conv2D(64, 3, strides=(2, 2), padding='same'),
        # keras.layers.Dropout(0.3),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding='same'),
        # keras.layers.Dropout(0.3),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(128, 3, strides=(2, 2), padding='same'),
        # keras.layers.Dropout(0.3),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(128, 3, strides=(1, 1), padding='same'),
        # keras.layers.Dropout(0.3),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(256, 3, strides=(2, 2), padding='same'),
        # keras.layers.Dropout(0.3),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(256, 3, strides=(1, 1), padding='same'),
        # keras.layers.Dropout(0.3),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(512, 3, strides=(2, 2), padding='same'),
        # keras.layers.Dropout(0.3),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(512, 3, strides=(1, 1), padding='same'),
        # keras.layers.Dropout(0.3),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Flatten(),

        keras.layers.Dense(64),
        tf.keras.layers.LeakyReLU(),

        keras.layers.Dense(1, activation='sigmoid'),
    ])

    optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
    optimizer_d = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

    generator_net.compile(optimizer=optimizer_g, loss='binary_crossentropy')
    discriminator_net.compile(optimizer=optimizer_d, loss='binary_crossentropy')

    generator_net.summary()
    discriminator_net.summary()

    return generator_net, discriminator_net

def generate_image():
    global generator_net
    if generator_net is None:
        raise ValueError("Generator network is not initialized. Call main() first.")
    noise = np.random.normal(0, 1, (1, 128))

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

    noise = np.random.normal(0, 1, (batch_size, 128))
    fake_imgs = generator_net.predict(noise, verbose=1)
    factor = max(fake_imgs.max(), abs(fake_imgs.min()))
    # fake_imgs = fake_imgs / factor

    print("After normalization:", images.min(), images.max())
    print("Real imgs min/max:", real_imgs.min(), real_imgs.max())
    print("Fake imgs min/max:", fake_imgs.min(), fake_imgs.max())

    d_loss_real = discriminator_net.train_on_batch(real_imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator_net.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
    d_loss = 0.5 * (d_loss_real + d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, 128))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    d_losses.append(d_loss)
    g_losses.append(g_loss)

    return d_loss, g_loss

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

def main():
    global gan, generator_net, discriminator_net
    get_global_variables()
    generator_net, discriminator_net = get_models()
    gan_input = tf.keras.layers.Input(shape=(128,))
    gan_output = discriminator_net(generator_net(gan_input))
    gan = tf.keras.Model(gan_input, gan_output)
    print("Variables are defined")
    gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5), loss='binary_crossentropy')

if __name__ == '__main__':
    main()