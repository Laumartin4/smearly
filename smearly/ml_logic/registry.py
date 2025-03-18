import os
import time

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

from smearly.params import *


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"models/model_{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/model_{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Create "models" dir if doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save model locally
    model_path = os.path.join("models", f"model_{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_SAVING_MODE == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/model_{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None


def load_model(local_model_path: str = 'model/model.h5') -> keras.Model:
    """
    Return a saved model:
    - locally form a "models" dir, using `model_filename`
    - or from GCS (most recent one) if MODEL_LOADING_MODE=='gcs'

    Return None or raise if no model is found

    """

    if MODEL_LOADING_MODE == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        if not os.path.isfile(local_model_path):
            raise FileNotFoundError(local_model_path, 'not found.')

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = keras.models.load_model(local_model_path)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_LOADING_MODE == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None
