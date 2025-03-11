from typing import Any
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def resize_pad_image_tf(image: Any, target_size: tuple[int,int]=(224, 224)) -> tf.Tensor:
    """
    Preprocesses an image by resizing, padding (with black), and normalizing it while maintaining the aspect ratio.

    Args:
    - image: A Tensor of type string. 0-D. The PNG-encoded image.
    - target_size (tuple): Target size (height, width) of the output image (default is 224x224).

    The image function parameter can be produced with this code:
    ```
    image_file_to_tf(image_path)
    ```

    Returns:
    - image (tf.Tensor): The preprocessed image ready for model input.
    """

    # Get original dimensions
    orig_height, orig_width = image.shape[:2]

    # Calculate the new size while maintaining the aspect ratio
    if orig_width > orig_height:
        new_width = target_size[0]
        new_height = (orig_height * target_size[0]) // orig_width
    else:
        new_height = target_size[1]
        new_width = (orig_width * target_size[1]) // orig_height

    # Resize the image to the new dimensions (maintaining aspect ratio)
    img_resized = tf.image.resize(image, (new_height, new_width), method='bilinear')

    # Create a padded image (black padding) of target_size
    img_padded = tf.image.resize_with_crop_or_pad(img_resized, target_size[0], target_size[1])

    # Normalize the image to [0, 1]
    img_normalized = tf.cast(img_padded, tf.float32) / 255.0

    return img_normalized


def image_file_to_tf(image_path: str) -> tf.Tensor:
    """
    Preprocesses an image by resizing, padding (with black), and normalizing it while maintaining the aspect ratio.

    Args:
    - image_path: Path to the image file (must be PNG but JPG and GIF are supported as well).

    Returns:
    - image (tf.Tensor): The image as a tensor of type tf.uint8 with 3 channels.
    """


    # Read the image file
    image = tf.io.read_file(image_path)

    # Decode the image into a tensor with 3 channels (RGB)
    return tf.io.decode_png(image, channels=3)


def create_image_dataset(directory: str, batch_size=32, target_size=(224, 224), shuffle=True) -> tf.data.Dataset:
    """
    Uses image_dataset_from_directory from tf to generate a tf.data.Dataset
    from image files in a directory. The directory must contain subdirectories
    named after the class and which contains the image files for each category.

    Args:
    - directory: Path to the parent directory (containing subdirs)
      Can be '../raw_data' (local) or 'gs://the-bucket-name/path-to-images-dir'

    Returns:
    - a tf.data.Dataset that can be used in a model.fit(the_ds)
    """
    return image_dataset_from_directory(
        directory,
        image_size=target_size,
        batch_size=batch_size,
        labels='inferred',
        label_mode='categorical',
        shuffle=shuffle,
        seed=42,
        pad_to_aspect_ratio=True
    )
