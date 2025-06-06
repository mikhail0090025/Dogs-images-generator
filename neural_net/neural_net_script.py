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
from torchvision import transforms

# Глобальные переменные
noise_size = 512
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

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

class GeneratorModel(nn.Module):
    def __init__(self, noise_size=noise_size):
        super(GeneratorModel, self).__init__()
        self.noise_size = noise_size
        relu_alpha = 0.2

        self.all_layers = nn.ModuleList([
            # Начало: шум → 2×2×512
            nn.Linear(noise_size, 2 * 2 * 256, bias=False),
            nn.LeakyReLU(relu_alpha),
            nn.Unflatten(1, (256, 2, 2)),

            # 2×2 → 4×4
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(256, 128),
            # nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_alpha),

            # 4×4 → 8×8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_alpha),
            SelfAttention(64),

            # 8×8 → 16×16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_alpha),
            SelfAttention(64),

            # 16×16 → 32×32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_alpha),
            SelfAttention(64),

            # 32×32 → 64×64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_alpha),

            # Финальный слой сглаживания
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Conv2d(3, 3, 1),
            nn.Tanh(),
        ])

        for layer in self.all_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)
            if isinstance(layer, nn.Upsample):
                noise = torch.randn_like(x) * 0.2
                x = x + noise
        return x

class DiscriminatorModel(nn.Module):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        relu_alpha = 0.2

        self.all_layers = nn.ModuleList([
            # 64×64 → 32×32
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(3, 32, 3, padding=1, stride=2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(relu_alpha),

            torch.nn.utils.spectral_norm(torch.nn.Conv2d(32, 32, 3, padding=1, stride=1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(relu_alpha),

            # 32×32 → 16×16
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(32, 64, 3, padding=1, stride=2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_alpha),

            torch.nn.utils.spectral_norm(torch.nn.Conv2d(64, 64, 3, padding=1, stride=1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_alpha),

            # 16×16 → 8×8
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(64, 128, 3, padding=1, stride=2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_alpha),

            torch.nn.utils.spectral_norm(torch.nn.Conv2d(128, 128, 3, padding=1, stride=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_alpha),

            # 8×8 → 8×8 (stride=1 для деталей)
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(128, 256, 3, padding=1, stride=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(relu_alpha),

            # Flatten и выход
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.LeakyReLU(relu_alpha),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)
        return x

from torch.utils.data import Dataset, DataLoader

# class DogsDataset(Dataset):
#     def __init__(self, images):
#         self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
# 
#     def __len__(self):
#         return len(self.images)
# 
#     def __getitem__(self, idx):
#         return self.images[idx]

class DogsDataset(Dataset):
    def __init__(self, images):
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
        
        # Определяем трансформации для аугментации
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(1),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # if self.transform:
        #     image = self.transform(image)
        return image

def get_models():
    global discriminator_net, generator_net, optimizer_G, optimizer_D
    generator_net = GeneratorModel().to(device)
    discriminator_net = DiscriminatorModel().to(device)

    optimizer_G = torch.optim.Adam(generator_net.parameters(), lr=0.00005, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator_net.parameters(), lr=0.00001, betas=(0.5, 0.999))

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

        generated = generated.squeeze(0)  # (1, 3, 64, 64) → (3, 64, 64)
        generated = generated.permute(1, 2, 0).cpu().numpy()  # (3, 64, 64) → (64, 64, 3)

    return generated

def train_one_epoch(generator_net, discriminator_net, optimizer_G, optimizer_D, dataloader, criterion):
    global d_losses, g_losses

    g_loss = train_one_epoch_generator(generator_net, discriminator_net, optimizer_G, dataloader, criterion)
    d_loss = train_one_epoch_discriminator(generator_net, discriminator_net, optimizer_D, dataloader, criterion)

    # d_losses.append(d_loss.item())
    # g_losses.append(g_loss.item())

    return d_loss.item(), g_loss.item()

def train_one_epoch_generator(generator_net, discriminator_net, optimizer_G, dataloader, criterion, add_last_values = False):
    global g_losses, d_losses
    for i, real_images in enumerate(dataloader):
        for param in discriminator_net.parameters():
            param.requires_grad_(False)
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

        for param in discriminator_net.parameters():
            param.requires_grad_(True)

        print(f"Batch {i+1}/{len(dataloader)}  G loss: {g_loss.item()}")

        g_losses.append(g_loss.item())
        if add_last_values:
            d_losses.append(d_losses[-1])

    return g_loss

def train_one_epoch_discriminator(generator_net, discriminator_net, optimizer_D, dataloader, criterion, add_last_values = False):
    global d_losses
    for i, real_images in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        for param in generator_net.parameters():
            param.requires_grad_(False)

        # Обучение только дискриминатора
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        output_real = discriminator_net(real_images)
        d_loss_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator_net(noise)
        fake_labels = torch.full((batch_size, 1), 0.0, device=device)
        output_fake = discriminator_net(fake_images.detach())
        d_loss_fake = criterion(output_fake, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        for param in generator_net.parameters():
            param.requires_grad_(True)
        
        print(f"Batch {i+1}/{len(dataloader)}  D loss: {d_loss.item()}")

        d_losses.append(d_loss.item())
        if add_last_values:
            g_losses.append(g_losses[-1])

    return d_loss

def train_one_epoch_by_batches(generator_net, discriminator_net, optimizer_D, dataloader, criterion):
    global d_losses, g_losses
    for i, real_images in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        for param in generator_net.parameters():
            param.requires_grad_(False)

        # Обучение только дискриминатора
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        output_real = discriminator_net(real_images)
        d_loss_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator_net(noise)
        fake_labels = torch.full((batch_size, 1), 0.0, device=device)
        output_fake = discriminator_net(fake_images.detach())
        d_loss_fake = criterion(output_fake, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        for param in generator_net.parameters():
            param.requires_grad_(True)

        for param in discriminator_net.parameters():
            param.requires_grad_(False)
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

        for param in discriminator_net.parameters():
            param.requires_grad_(True)

        print(f"Batch {i+1}/{len(dataloader)}  D loss: {d_loss.item()}  G loss: {g_loss.item()}")

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    return d_loss, g_loss


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

def train_epochs_by_batches(epochs_count, batch_size, generator_net, discriminator_net, optimizer_G, optimizer_D):
    global images
    if images is None:
        raise ValueError("Images are not loaded. Call get_global_variables() first.")
    
    dataset = DogsDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()

    for epoch in range(epochs_count):
        train_one_epoch_by_batches(generator_net, discriminator_net, optimizer_D, dataloader, criterion)

    print(f"{epochs_count} epochs passed!")

def train_epochs_generator(epochs_count, batch_size, generator_net, discriminator_net, optimizer_G, add_last_values = False):
    global images
    if images is None:
        raise ValueError("Images are not loaded. Call get_global_variables() first.")
    
    dataset = DogsDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()

    for epoch in range(epochs_count):
        g_loss = train_one_epoch_generator(generator_net, discriminator_net, optimizer_G, dataloader, criterion, add_last_values)
        print(f"Epoch {epoch+1}/{epochs_count}: G Loss: {g_loss:.4f}")

    print(f"{epochs_count} epochs passed!")

def train_epochs_discriminator(epochs_count, batch_size, generator_net, discriminator_net, optimizer_D, add_last_values = False):
    global images
    if images is None:
        raise ValueError("Images are not loaded. Call get_global_variables() first.")
    
    dataset = DogsDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()

    for epoch in range(epochs_count):
        d_loss = train_one_epoch_discriminator(generator_net, discriminator_net, optimizer_D, dataloader, criterion, add_last_values)
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