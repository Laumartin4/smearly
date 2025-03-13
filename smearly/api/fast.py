
import pandas as pd
#IMPORT d'une fonction type load_model
#IMPORT du preprocessing (choisir les fonctions)
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.preprocessing import image_file_to_tf, create_image_dataset, resize_pad_image_tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app.state.model = #load_model ou autre fonction qui fait tourner le modèle

model = app.state.model


@app.post('/upload_file')
async def upload_file(file: UploadFile | None = None):
    if not file :
        return f'No file uploaded'
    else :
        return f'Your file named {file.filename} has been successfully loaded'


def preproc_file_input(file):
    input_file_preprocessed = image_file_to_tf(file_path)
    return input_file_preprocessed


@app.get('/predict')
async def predict(input_file_preprocessed):
    prediction = model.predict(input_file_preprocessed)
    return {'healthy' : prediction['healthy'],
            'unhealthy' : prediction['unhealthy'],
            'rubbish' : prediction['rubbish']}

    # if (model.predict['healthy'] > model.predict['unhealthy']) and
    # (model.predict['healthy'] > model.predict['rubbish']) :
    #     f'The picture shows healthy cells'

    # elif (model.predict['unhealthy'] > model.predict['healthy']) and
    # (model.predict['unhealthy'] > model.predict['rubbish']) :
    #     f 'The picture shows unhealthy cells'

    # else :
    #     f 'The picture cannot be interpreted'
