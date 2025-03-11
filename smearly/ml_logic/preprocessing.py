import tensorflow as tf


def resize_pad_image_tf(image_path: str, target_size: tuple[int,int]=(224, 224)) -> tf.Tensor:
    """
    Preprocesses an image by resizing, padding (with black), and normalizing it while maintaining the aspect ratio.

    Args:
    - image_path (str): Path to the image file.
    - target_size (tuple): Target size (height, width) of the output image (default is 224x224).

    Returns:
    - img (tf.Tensor): The preprocessed image ready for model input.
    """
    # Read the image file
    img = tf.io.read_file(image_path)

    # Decode the image into a tensor with 3 channels (RGB)
    img = tf.image.decode_png(img, channels=3)

    # Get original dimensions
    orig_height, orig_width = img.shape[:2]

    # Calculate the new size while maintaining the aspect ratio
    if orig_width > orig_height:
        new_width = target_size[0]
        new_height = (orig_height * target_size[0]) // orig_width
    else:
        new_height = target_size[1]
        new_width = (orig_width * target_size[1]) // orig_height

    # Resize the image to the new dimensions (maintaining aspect ratio)
    img_resized = tf.image.resize(img, (new_height, new_width), method='bilinear')

    # Create a padded image (black padding) of target_size
    img_padded = tf.image.resize_with_crop_or_pad(img_resized, target_size[0], target_size[1])

    # Normalize the image to [0, 1]
    img_normalized = tf.cast(img_padded, tf.float32) / 255.0

    return img_normalized
