
from click import File
import pandas as pd
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from smearly.ml_logic.preprocessing import resize_pad_image_tf
from smearly.ml_logic.model import load_model
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0

from smearly.params import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app.state.model = load_model(MODEL_FILENAME)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        print("Received image for prediction")

        # Read and open the image
        image = Image.open(io.BytesIO(await file.read()))
        image = image.convert("RGB")  # Ensure it's RGB (3 channels)
        image_array = np.asarray(image)

        # Convert NumPy array to TensorFlow Tensor
        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.uint8)

        # Resize and preprocess the image
        resized_image = resize_pad_image_tf(image_tensor, normalize=False)  # Ensure (224, 224, 3)
        image_batch = tf.expand_dims(resized_image, axis=0)  # Add batch dimension (1, 224, 224, 3)

        # Make a prediction
        prediction = app.state.model.predict(image_batch)
        prediction = prediction.squeeze()

        # Return the prediction
        print(f"Prediction type: {type(prediction)}, shape: {prediction.shape}")

        dict_output = {'healthy': float(prediction[0]),
                       'rubbish': float(prediction[1]),
                       'unhealthy': float(prediction[2])}

        max_key = max(dict_output, key=dict_output.get)

        return JSONResponse(content={"prediction": dict_output,
                                     'main class': max_key})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
