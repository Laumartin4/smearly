import os

# defaults to local model saving & loading
MODEL_SAVING_MODE = os.environ.get("MODEL_SAVING_MODE", "local")
MODEL_LOADING_MODE = os.environ.get("MODEL_LOADING_MODE", "local")
