import pandas as pd
from tensorflow.keras.utils import to_categorical

def encode_labels(y):
    """
    Encode labels to category encoding
    :param y: labels
    """
    y_cat = to_categorical(y)
    return y_cat




