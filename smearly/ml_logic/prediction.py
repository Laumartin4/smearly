import pandas as pd
from tensorflow.keras.utils import to_categorical

def encode_labels(y):
    """
    Encode labels to category encoding
    :param y: labels
    """
    y_cat = to_categorical(y)
    return y_cat


def prediction_to_csv(y_pred : pd.DataFrame, filename : str = "predictions.csv"):
    """
    Save predictions to CSV file
    """
    y_pred = y_pred.argmax(axis=1)
    df = pd.DataFrame(y_pred, columns=["label"])
    df.to_csv(filename, index=False)
    print(f"✅ Predictions saved to {filename}")

