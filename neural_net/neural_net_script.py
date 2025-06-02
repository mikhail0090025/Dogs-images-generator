import numpy as np
import requests
import io
import tempfile
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim

# Глобальные переменные
noise_size = 128
generator_net = None
discriminator_net = None
optimizer_D = None
optimizer_G = None
gan = None
images = None
images_shape = None
d_losses = []
g_losses = []
print("Variables are reset")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class GeneratorModel(nn.Module):
    def __init__(self, noise_size=noise_size):
        super(GeneratorModel, self).__init__()
        self.noise_size = noise_size
        '''
        self.all_layers = nn.ModuleList([
            nn.Linear(noise_size, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 2 * 2 * 512),
            nn.LeakyReLU(0.2),

            nn.Linear(2 * 2 * 512, 2 * 2 * 512),
            nn.LeakyReLU(0.2),

            nn.Unflatten(1, (512, 2, 2)),

            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 2x2 → 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 3, padding=1),  # 4x4 → 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4x4 → 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, 3, padding=1),  # 8x8 → 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8 → 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, 3, padding=1),  # 16x16 → 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16x16 → 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, 3, padding=1),  # 32x32 → 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32x32 → 64x64
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 16, 3, padding=1),  # 64x64 → 64x64
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 3, 3, padding=1),  # 64x64 → 64x64x3
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        ])
        '''
        self.all_layers = nn.ModuleList([
            nn.Linear(noise_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 2 * 2 * 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(2 * 2 * 1024, 2 * 2 * 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Unflatten(1, (1024, 2, 2)),

            nn.Conv2d(1024, 1024, 3, padding=1),  # 2x2 → 2x2
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, 3, padding=1),  # 4x4 → 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, 3, padding=1),  # 4x4 → 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 3, padding=1),  # 8x8 → 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, 3, padding=1),  # 16x16 → 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, 3, padding=1),  # 32x32 → 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, 3, padding=1),  # 64x64 → 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 3, 3, padding=1),  # 64x64 → 64x64x3
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)
        return x

class DiscriminatorModel(nn.Module):

    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        self.all_layers = nn.ModuleList([
            torch.nn.Conv2d(3, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            torch.nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            torch.nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            torch.nn.Conv2d(256, 512, 3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            torch.nn.Conv2d(512, 1024, 3, padding=1, stride=2),
            nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Flatten(),

            nn.Linear(4096, 1024),
            torch.nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            torch.nn.LeakyReLU(0.2),

            nn.Linear(512, 64),
            torch.nn.LeakyReLU(0.2),

            nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        ])

    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)
        return x

from torch.utils.data import Dataset, DataLoader

class DogsDataset(Dataset):
    def __init__(self, images):
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

def get_models():
    global discriminator_net, generator_net, optimizer_G, optimizer_D
    generator_net = GeneratorModel().to(device)
    discriminator_net = DiscriminatorModel().to(device)

    optimizer_G = torch.optim.Adam(generator_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print("Generator model structure:")
    print(generator_net)
    print("Discriminator model structure:")
    print(discriminator_net)

    return generator_net, discriminator_net, optimizer_G, optimizer_D

def generate_image():
    global generator_net
    if generator_net is None:
        raise ValueError("Generator network is not initialized. Call main() first.")
    noise = torch.randn(1, noise_size).to(device)

    with torch.no_grad():
        generated = generator_net(noise)
        print("Generated shape:", generated.shape)  # Expecting (1, 3, 64, 64)
        print("Data min/max:", generated.min().item(), generated.max().item())

        if generated.min().item() < 0:
            generated = (generated + 1) / 2.0
        generated = torch.clamp(generated, 0, 1)

        generated = generated.squeeze(0)  # (1, 3, 64, 64) → (3, 64, 64)
        generated = generated.permute(1, 2, 0).cpu().numpy()  # (3, 64, 64) → (64, 64, 3)

    return generated

def train_one_epoch(generator_net, discriminator_net, optimizer_G, optimizer_D, dataloader, criterion):
    global d_losses, g_losses
    for real_images in dataloader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # Обучение дискриминатора
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # Label smoothing
        output_real = discriminator_net(real_images)
        d_loss_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator_net(noise)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        output_fake = discriminator_net(fake_images.detach())
        d_loss_fake = criterion(output_fake, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # Обучение генератора
        optimizer_G.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)  # Хотим, чтобы фейковые изображения казались настоящими
        output = discriminator_net(fake_images)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_G.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    return d_loss.item(), g_loss.item()

def train_one_epoch_generator(generator_net, discriminator_net, optimizer_G, dataloader, criterion):
    global g_losses
    for real_images in dataloader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # Обучение только генератора
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator_net(noise)
        real_labels = torch.ones(batch_size, 1).to(device)
        output = discriminator_net(fake_images)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_G.step()

        g_losses.append(g_loss.item())

    return g_loss.item()

def train_one_epoch_discriminator(generator_net, discriminator_net, optimizer_D, dataloader, criterion):
    global d_losses
    for real_images in dataloader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # Обучение только дискриминатора
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # Label smoothing
        output_real = discriminator_net(real_images)
        d_loss_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator_net(noise)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        output_fake = discriminator_net(fake_images.detach())
        d_loss_fake = criterion(output_fake, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        d_losses.append(d_loss.item())

    return d_loss.item()

def train_epochs(epochs_count, batch_size, generator_net, discriminator_net, optimizer_G, optimizer_D):
    global images
    if images is None:
        raise ValueError("Images are not loaded. Call get_global_variables() first.")
    
    dataset = DogsDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()

    for epoch in range(epochs_count):
        d_loss, g_loss = train_one_epoch(generator_net, discriminator_net, optimizer_G, optimizer_D, dataloader, criterion)
        print(f"Epoch {epoch+1}/{epochs_count}: D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

    print(f"{epochs_count} epochs passed!")

def train_epochs_generator(epochs_count, batch_size, generator_net, discriminator_net, optimizer_G):
    global images
    if images is None:
        raise ValueError("Images are not loaded. Call get_global_variables() first.")
    
    dataset = DogsDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()

    for epoch in range(epochs_count):
        g_loss = train_one_epoch_generator(generator_net, discriminator_net, optimizer_G, dataloader, criterion)
        print(f"Epoch {epoch+1}/{epochs_count}: G Loss: {g_loss:.4f}")

    print(f"{epochs_count} epochs passed!")

def train_epochs_discriminator(epochs_count, batch_size, generator_net, discriminator_net, optimizer_D):
    global images
    if images is None:
        raise ValueError("Images are not loaded. Call get_global_variables() first.")
    
    dataset = DogsDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()

    for epoch in range(epochs_count):
        d_loss = train_one_epoch_discriminator(generator_net, discriminator_net, optimizer_D, dataloader, criterion)
        print(f"Epoch {epoch+1}/{epochs_count}: D Loss: {d_loss:.4f}")

    print(f"{epochs_count} epochs passed!")

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

def main():
    global gan, generator_net, discriminator_net, optimizer_G, optimizer_D
    get_global_variables()
    generator_net, discriminator_net, optimizer_G, optimizer_D = get_models()

if __name__ == '__main__':
    main()