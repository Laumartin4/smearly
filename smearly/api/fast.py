
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.preprocessing import resize_pad_image_tf
from ml_logic.model import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



model = load_model("model_064")


@app.post('/predict')
async def predict(file: UploadFile = File(...)):

     try:
        # Lire et ouvrir l'image envoyée
        image = Image.open(io.BytesIO(await file.read()))

        # Prétraiter l'image en utilisant les fonctions importées
        resized_image = resize_pad_image_tf(image)  # Redimensionner et pad l'image

        # Effectuer la prédiction
        prediction = model.predict(resized_image)

        # Retourner la prédiction sous forme de réponse JSON
        return JSONResponse(content={"prediction": prediction.tolist()})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
