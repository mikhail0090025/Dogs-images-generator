from fastapi import FastAPI
from fastapi.responses import Response, FileResponse, RedirectResponse
import os
import asyncio
import neural_net_script
import io
from PIL import Image
import numpy as np

app = FastAPI()

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
        return Response(f"Error has occurred: {e}", status_code=500)