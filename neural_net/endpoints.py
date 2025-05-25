from fastapi import FastAPI, Form
from fastapi.responses import Response, FileResponse, RedirectResponse
import os
import asyncio
import neural_net_script
import io
from PIL import Image
import numpy as np

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
        asyncio.run(neural_net_script.go_epochs(int(epochs_count), int(batch_size)))
        return Response(f"{epochs_count} Epochs were passed", status_code=200)
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

if __name__ == '__main__':
    neural_net_script.main()