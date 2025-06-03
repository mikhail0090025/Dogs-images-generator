from fastapi import FastAPI, Form
from fastapi.responses import Response, FileResponse, RedirectResponse
import os
import asyncio
import neural_net_script
import io
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import requests

app = FastAPI()

# CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    neural_net_script.main()

@app.get("/")
def root():
    return "This is a root of neural net microservice"

@app.get("/generate")
async def generate_endpoint():
    try:
        # Генерируем изображение
        generated = neural_net_script.generate_image()

        # Преобразуем numpy-массив в изображение
        generated = (generated * 255).astype(np.uint8)  # Преобразуем в [0, 255] и uint8
        img = Image.fromarray(generated)

        # Сохраняем изображение в байты
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Возвращаем изображение как байты
        return Response(content=img_byte_arr, media_type="image/png")
    except Exception as e:
        print(f"Error has occurred: {e}")
        return Response(f"Error has occurred: {e}", status_code=500)

@app.post("/pass_epochs")
def pass_epochs_endpoint(epochs_count, batch_size):
    try:
        if not epochs_count:
            return Response({'error': 'No epochs_count provided'}, status_code=400)
        if not batch_size:
            return Response({'error': 'No batch_size provided'}, status_code=400)

        print(f"Epochs count: {epochs_count}")
        print(f"Batch size: {batch_size}")
        asyncio.run(neural_net_script.train_epochs(int(epochs_count), int(batch_size), neural_net_script.generator_net, neural_net_script.discriminator_net, neural_net_script.optimizer_G, neural_net_script.optimizer_D))
        return Response(f"{epochs_count} Epochs were passed", status_code=200)
    except Exception as e:
        print(f"Error has occurred: {e}")
        return Response(f"Error has occurred: {e}", status_code=500)

@app.post("/pass_epochs_by_batches")
def pass_epochs_endpoint(epochs_count, batch_size):
    try:
        if not epochs_count:
            return Response({'error': 'No epochs_count provided'}, status_code=400)
        if not batch_size:
            return Response({'error': 'No batch_size provided'}, status_code=400)

        print(f"Epochs count: {epochs_count}")
        print(f"Batch size: {batch_size}")
        asyncio.run(neural_net_script.train_epochs_by_batches(int(epochs_count), int(batch_size), neural_net_script.generator_net, neural_net_script.discriminator_net, neural_net_script.optimizer_G, neural_net_script.optimizer_D))
        return Response(f"{epochs_count} Epochs by batches were passed", status_code=200)
    except Exception as e:
        print(f"Error has occurred: {e}")
        return Response(f"Error has occurred: {e}", status_code=500)

@app.post("/pass_epochs_discriminator")
def pass_epochs_discriminator_endpoint(epochs_count, batch_size):
    try:
        if not epochs_count:
            return Response({'error': 'No epochs_count provided'}, status_code=400)
        if not batch_size:
            return Response({'error': 'No batch_size provided'}, status_code=400)

        print(f"Epochs count: {epochs_count}")
        print(f"Batch size: {batch_size}")
        asyncio.run(neural_net_script.train_epochs_discriminator(int(epochs_count), int(batch_size), neural_net_script.generator_net, neural_net_script.discriminator_net, neural_net_script.optimizer_D, True))
        return Response(f"{epochs_count} Epochs for discriminator were passed", status_code=200)
    except Exception as e:
        print(f"Error has occurred: {e}")
        return Response(f"Error has occurred: {e}", status_code=500)

@app.post("/pass_epochs_generator")
def pass_epochs_discriminator_endpoint(epochs_count, batch_size):
    try:
        if not epochs_count:
            return Response({'error': 'No epochs_count provided'}, status_code=400)
        if not batch_size:
            return Response({'error': 'No batch_size provided'}, status_code=400)

        print(f"Epochs count: {epochs_count}")
        print(f"Batch size: {batch_size}")
        asyncio.run(neural_net_script.train_epochs_generator(int(epochs_count), int(batch_size), neural_net_script.generator_net, neural_net_script.discriminator_net, neural_net_script.optimizer_G, True))
        return Response(f"{epochs_count} Epochs for generator were passed", status_code=200)
    except Exception as e:
        print(f"Error has occurred: {e}")
        return Response(f"Error has occurred: {e}", status_code=500)

@app.get('/graphic')
def graphic():
    try:
        fig = neural_net_script.plot_losses()
        fig_json = fig.to_json()
        return {
            'fig_json': fig_json,
        }
    except Exception as e:
        return {'Response': f'Unexpected error has occured: {e}, ({type(e)})'}, 500

@app.post("/predict_discriminator")
def predict_discriminator_endpoint(image_url: str = Form(...)):
    try:
        # Проверяем, что URL передан
        print(f"Fetching image from URL: {image_url}")
        if not image_url:
            return Response("No image URL provided", status_code=400)

        # Загружаем изображение по URL
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch image from URL: {image_url}, status code: {response.status_code}")
            return Response(f"Failed to fetch image from URL: {image_url}", status_code=400)

        # Открываем изображение из байтов
        img = Image.open(io.BytesIO(response.content)).convert('RGB')  # Конвертируем в RGB

        # Подготавливаем трансформации (как для тренировочных данных)
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Предполагаю, что вход дискриминатора 64x64
            transforms.ToTensor(),  # Конвертируем в тензор [0, 1]
        ])
        img_tensor = transform(img).unsqueeze(0)  # Добавляем размерность батча: [1, 3, 64, 64]

        # Переносим на устройство
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.to(device)

        # Пропускаем через дискриминатор
        with torch.no_grad():  # Отключаем градиенты для инференса
            prediction = neural_net_script.discriminator_net(img_tensor)
            prediction = torch.sigmoid(prediction).item()  # Применяем сигмоиду и берём скаляр

        # Возвращаем предсказание (0 - фейк, 1 - реальное)
        return {
            "image_url": image_url,
            "discriminator_prediction": prediction,
            "interpretation": "Real" if prediction > 0.5 else "Fake"
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from URL: {e}")
        return Response(f"Error fetching image from URL: {e}", status_code=400)
    except Exception as e:
        print(f"Error has occurred: {e}")
        return Response(f"Error has occurred: {e}", status_code=500)

if __name__ == '__main__':
    neural_net_script.main()