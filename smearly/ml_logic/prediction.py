import os
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image_dataset_from_directory

from smearly.ml_logic.registry import load_model
from smearly.ml_logic.preprocessing import image_file_to_tf, resize_pad_image_tf
from smearly.tools.submission import prediction_to_csv


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



def predict_for_kaggle_without_tf_ds(
        model_filename: str='../models/model-064.h5',
        csv_file_to_predict: str='../csv_files/isbi2025-ps3c-test-dataset.csv',
        test_images_dir: str='../raw_data/test_ds',
        chunk_size: int=32,
        normalize: bool=False
    ):
    """
    Batch predict all images from `csv_file_to_predict`, getting images from
    `test_images_dir` and using model from `model_filename`.

    Does a "manual" resize of images with resize_pad_image_tf().

    Generates a CSV file with prediction_to_csv() function above.
    """

    model = load_model(model_filename)

    predictions = np.empty((0, 3))
    for chunk in pd.read_csv(csv_file_to_predict, chunksize=chunk_size):
        batch_img_list = []
        for image_filename in chunk['image_name']:
            img = resize_pad_image_tf(image_file_to_tf(os.path.join(test_images_dir, image_filename)), normalize=normalize)
            # expand_dims() not needed if tf.stack is used to put several images in a single tensor
            #img = tf.expand_dims(img, axis=0)
            batch_img_list.append(img)

        batch_tensor = tf.stack(batch_img_list, axis=0)
        batch_predictions = model.predict(batch_tensor)

        predictions = np.vstack((predictions, batch_predictions))

        date_time = dt.datetime.now().strftime('%Y%m%d_%H%M')

    prediction_to_csv(predictions, dest_filename=f'test_predictions_{date_time}.csv', src_filename=csv_file_to_predict)
