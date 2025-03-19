import os

# defaults to local model saving & loading
MODEL_SAVING_MODE = os.environ.get("MODEL_SAVING_MODE", "local")
MODEL_LOADING_MODE = os.environ.get("MODEL_LOADING_MODE", "local")

MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "models/model-064.h5")
