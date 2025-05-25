from fastapi import FastAPI
from fastapi.responses import Response, FileResponse, RedirectResponse
from download_dogs_images import get_images
import os
from main_variables import images_shape, images_resolution, current_dir, file_name, dataset_directory
import asyncio

app = FastAPI()

@app.get("/")
def root():
    return RedirectResponse('/get_images')

@app.get("/get_images")
def get_images_endpoint():
    try:
        if os.path.exists(os.path.join(current_dir, 'numpy_dataset.npz')):
            return FileResponse('numpy_dataset.npz', status_code=200)
        else:
            asyncio.run(get_images())
            return FileResponse('numpy_dataset.npz', status_code=200)
    except Exception as e:
        return Response(f'Unexpected error has occured: {e}', status_code=500)

@app.get("/images_shape")
def get_images_shape():
    try:
        return images_shape
    except Exception as e:
        return Response(f'Unexpected error has occured: {e}', status_code=500)
