from typing import Any
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def resize_pad_image_tf(image: Any, target_size: tuple[int,int] | None=(224, 224), normalize: bool=True) -> tf.Tensor:
    """
    Preprocesses an image by resizing, padding (with black), and normalizing it while maintaining the aspect ratio.

    Args:
    - image: A Tensor of type string. 0-D. The PNG-encoded image. Hint: use image_file_to_tf() to generate this.
    - target_size (tuple): Target size (height, width) of the output image (default is 224x224) or None to skip resizing.
    - normalize (bool): pixel values will be divided by 255 if True, untouched otherwise

    The image function parameter can be produced with this code:
    ```
    image_file_to_tf(image_path)
    ```

    Returns:
    - image (tf.Tensor): The preprocessed image ready for model input.
    """
    if target_size is not None:
        # Resize the image to the target size while maintaining aspect ratio and pad if necessary
        image = tf.image.resize_with_pad(image, target_size[0], target_size[1], method='bilinear')

    if normalize is True:
        # Normalize the image to [0, 1]
        return tf.cast(image, tf.float32) / 255.0
    else:
        # the resize above with any method other than 'nearest' will produce floats so...
        # Ensure values are in the range [0, 255] (if they aren't already) and cnverted to uint8
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.cast(image, tf.uint8)

        return image


def image_file_to_tf(image_path: str) -> tf.Tensor:
    """
    Loads an image file from disk, and returns it as a tf.Tensor

    Args:
    - image_path: Path to the image file (must be PNG but JPG and GIF are supported as well).

    Returns:
    - image (tf.Tensor): The image as a tensor of type tf.uint8 with 3 channels.
    """


    # Read the image file
    image = tf.io.read_file(image_path)

    # Decode the image into a tensor with 3 channels (RGB)
    return tf.io.decode_png(image, channels=3)


def create_image_dataset(directory: str,
                         batch_size: int=32,
                         target_size: tuple[int,int] = (224, 224),
                         shuffle: bool = True,
                         seed: int = 42,
                         pad: bool = True,
                         normalize: bool=True) -> tf.data.Dataset:
    """
    Uses image_dataset_from_directory from tf to generate a tf.data.Dataset
    from image files in a directory. The directory must contain subdirectories
    named after the class and which contains the image files for each category.
    Resizing cannot be avoided (but will probably be a no-op if source size
    matches target size).

    Args:
    - directory: Path to the parent directory (containing subdirs)
      Can be '../raw_data' (local) or 'gs://the-bucket-name/path-to-images-dir'

    Returns:
    - a tf.data.Dataset that can be used in a model.fit(the_ds),
      with the pixel values normalized (values are float between 0 and 1)
    """
    def normalize_image(image, label):
        # Normalize the pixel values to the range [0, 1]
        return image/255.0, label

    img_ds = image_dataset_from_directory(
        directory,
        image_size=target_size,
        batch_size=batch_size,
        labels='inferred',
        label_mode='categorical',
        shuffle=shuffle,
        seed=seed,
        pad_to_aspect_ratio=pad
    )
    if normalize is True:
        return img_ds.map(normalize_image)
    else:
        return img_ds
