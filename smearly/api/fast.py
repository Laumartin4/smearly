
import pandas as pd
#IMPORT d'une fonction type load_model
#IMPORT du preprocessing (choisir les fonctions)
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from smearly.ml_logic.preprocessing import image_file_to_tf, create_image_dataset, resize_pad_image_tf



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# app.state.model = #load_model ou autre fonction qui fait tourner le modèle

# model = app.state.model

# def preproc_file_input(file_path):
#     input_file_preprocessed = image_file_to_tf(file_path)
#     return input_file_preprocessed

# preproc_file_input(file)

#clarifier quel type d'objet est retourné par le preprocessing


@app.post('/predict')
async def predict(request: Request):
    data = await request.body()
    print(data)
    # prediction = model.predict(data)
    # return {'healthy' : prediction['healthy'],
    #         'unhealthy' : prediction['unhealthy'],
    #         'rubbish' : prediction['rubbish']}


# @app.get('/')
# def root():
#     return {'reply':'toto'}


    # if (model.predict['healthy'] > model.predict['unhealthy']) and
    # (model.predict['healthy'] > model.predict['rubbish']) :
    #     f'The picture shows healthy cells'

    # elif (model.predict['unhealthy'] > model.predict['healthy']) and
    # (model.predict['unhealthy'] > model.predict['rubbish']) :
    #     f 'The picture shows unhealthy cells'

    # else :
    #     f 'The picture cannot be interpreted'
