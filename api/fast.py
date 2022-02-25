from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from utils import prediction, prediction2, prediction3
import numpy as np
from imutils import url_to_image
import uvicorn
import requests
import cv2
from PIL import Image, ImageFilter
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import io
from io import BytesIO
from starlette.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post('/randomgen')
def randomgen():
    url = 'https://randomuser.me/api/'
    response = requests.get(url).json()
    img_url = response['results'][0]['picture']['large']

    url_ar = url_to_image(img_url)

    #Image.open(img_url)

    return prediction2(url_ar)



@app.post('/predict/img')

def prediction_up():
    img= './img.jpg'
    return prediction(img)

@app.post('/upload')
def image_up(img: UploadFile = File(...)):
    og_image = Image.open(img.file)
    og_image = og_image.filter(ImageFilter.BLUR)

    fil_image = BytesIO()
    og_image.save(fil_image, 'JPEG')
    fil_image.seek(0)

    return StreamingResponse(fil_image, media_type = 'image/jpeg')
