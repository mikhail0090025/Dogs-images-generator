import os
import io
import asyncio
import aiohttp
import aiofiles
import zipfile
import numpy as np
from PIL import Image, UnidentifiedImageError
import shutil
from main_variables import images_shape, images_resolution, current_dir, file_name, dataset_directory

np.random.seed(42)
images_are_loaded = False
images = []

async def download_dataset(session, url, filepath):
    async with session.get(url) as response:
        if response.status == 200:
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(await response.read())
            print(f"Downloaded {filepath}")
        else:
            raise Exception(f"Download failed with status {response.status}")

async def extract_zip(zip_path, extract_to):
    async with aiofiles.open(zip_path, 'rb') as f:
        content = await f.read()
        with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
            zip_ref.extractall(extract_to)
    print("Dataset extracted to:", extract_to)

    # Удаляем папку с кошками
    cat_folder = os.path.join(dataset_directory, "Cat")
    if os.path.exists(cat_folder):
        shutil.rmtree(cat_folder)
        print(f"Removed folder: {cat_folder}")

async def process_images(folder):
    global images
    all_files = os.listdir(folder)
    max_files_count = 512
    for i, filename in enumerate(all_files[:max_files_count]):
        try:
            path = os.path.join(folder, filename)
            print(f"Path: {path}\n{i+1}/{max_files_count}")
            img = Image.open(path)
            img = img.convert("RGB")
            img_resized = img.resize(images_resolution, Image.Resampling.LANCZOS)
            img_array = (np.array(img_resized) / 127.5).astype(np.float32)
            img_array = img_array - 1
            images.append(img_array)
            print("Dog image")
        except UnidentifiedImageError:
            print(f"Error loading {path}: not an image, skipping")
            continue

async def get_images():
    global images, outputs, images_are_loaded

    try:
        if os.path.exists(os.path.join(current_dir, 'numpy_dataset.npz')):
            print("Dataset was found. Loading...")
            data = np.load(os.path.join(current_dir, 'numpy_dataset.npz'))
            images = data['images']
            images_are_loaded = True
            return

        print("Dataset was not found. Creating...")
        url = "https://www.kaggle.com/api/v1/datasets/download/bhavikjikadara/dog-and-cat-classification-dataset"
        zip_path = os.path.join(current_dir, file_name)

        if not os.path.exists(dataset_directory):
            if not os.path.exists(os.path.join(current_dir, file_name)):
                async with aiohttp.ClientSession() as session:
                    await download_dataset(session, url, zip_path)
            await extract_zip(zip_path, current_dir)

        dog_folder = os.path.join(dataset_directory, "Dog")
        cat_folder = os.path.join(dataset_directory, "Cat")
        if not os.path.exists(dog_folder):
            raise Exception(f"Dog folder not found: {dog_folder}")
        if not os.path.exists(cat_folder):
            raise Exception(f"Cat folder not found: {cat_folder}")

        await process_images(dog_folder)
        # await process_images(cat_folder)

        images = np.array(images, dtype=np.float32)
        indexes = np.random.permutation(len(images))
        images = images[indexes]
        images = np.array(images, dtype=np.float32)
        print(images)
        np.savez_compressed('numpy_dataset.npz', images=images)
        images_are_loaded = True
    except Exception as e:
        print(f'Error while getting dataset has occurred: {e}')
        raise

def bytesToText(bytes: int):
    if bytes < 1024:
        return f'{bytes} Bytes.'
    elif bytes < 1024 * 1024:
        return f'{bytes / 1024:.2f} KB.'
    elif bytes < 1024 * 1024 * 1024:
        return f'{bytes / (1024 * 1024):.2f} MB.'
    elif bytes < 1024 * 1024 * 1024 * 1024:
        return f'{bytes / (1024 * 1024 * 1024):.2f} GB.'
    else:
        return f'{bytes / (1024 * 1024 * 1024 * 1024):.2f} TB.'

if __name__ == '__main__':
    asyncio.run(get_images())
    print(f'Images count: {len(images)}')
    print(f'Dataset size: {bytesToText(images.size * 4)}.')
    print(f'Inputs shape: {images.shape}')