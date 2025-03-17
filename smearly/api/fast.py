
import pandas as pd
#IMPORT d'une fonction type load_model
#IMPORT du preprocessing (choisir les fonctions)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from smearly.ml_logic.preprocessing import image_file_to_tf, resize_pad_image_tf
from ml_logic.preprocessing import resize_pad_image_tf, image_file_to_tf




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


@app.post('/predict')
async def predict(file: UploadFile = File(...)):

     try:
        # Lire et ouvrir l'image envoyée
        image = Image.open(io.BytesIO(await file.read()))

        # Prétraiter l'image en utilisant les fonctions importées
        resized_image = resize_pad_image_tf(image)  # Redimensionner et pad l'image
        preprocessed_image = image_file_to_tf(resized_image)  # Convertir l'image pour TensorFlow

        # Effectuer la prédiction
        prediction = model.predict(preprocessed_image)

        # Retourner la prédiction sous forme de réponse JSON
        return JSONResponse(content={"prediction": prediction.tolist()})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
