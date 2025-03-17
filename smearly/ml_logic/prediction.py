import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image_dataset_from_directory


def create_image_dataset_test(directory: str,
                         batch_size: int=32,
                         target_size: tuple[int,int] = (224, 224),
                         pad: bool = True,
                         normalize: bool=True) -> tf.data.Dataset:
    """
    Uses image_dataset_from_directory from tf to generate a tf.data.Dataset
    from image files in a directory. 
    The directory must contain all the image files for test.
    Resizing cannot be avoided (but will probably be a no-op if source size
    matches target size).

    Args:
    - directory: Path to the parent directory 
      Can be '../raw_data' (local) or 'gs://the-bucket-name/path-to-images-dir'

    Returns:
    - a tf.data.Dataset that can be used in a model.predict(the_ds),
      with the pixel values normalized (values are float between 0 and 1)
    """

    def normalize_image(image):
        # Normalize the pixel values to the range [0, 1]
        return image/255.0
    
    img_ds = image_dataset_from_directory(
        directory,
        image_size=target_size,
        batch_size=batch_size,
        labels = None,
        shuffle= None,
        pad_to_aspect_ratio = pad
    )
    if normalize is True:
        return img_ds.map(normalize_image)  # Normalize the pixel values to the range [0, 1]
    else:
        return img_ds





