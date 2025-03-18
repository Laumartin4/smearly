import pandas as pd
import numpy as np

def prediction_to_csv(y_pred: np.ndarray, dest_filename: str="test_predictions.csv", src_filename: str='../raw_data/isbi2025-ps3c-test-dataset.csv'):
    """
    Save some predictions to a CSV file for Kaggle submission

    The number of predictions must match the number of lines (header excluded) in `src_filename`

    Args:

     - y_pred: a 2 dims np.ndarray of shape: (n, 3) containing `n` "softmax"
       predictions for the 3 classes healthy, rubbish and unhealthy (in that order)
       These predictions must be in ordered as in `src_filename`.

     - dest_filename: target CSV file (source CSV files with a `label` column added for predictions)

     - src_filename: usually, path to the Kaggle `isbi2025-ps3c-test-dataset.csv` submission file.
    """
    y_pred = y_pred.argmax(axis=1)

    y_test_df = pd.read_csv(src_filename)
    y_test_df["label"] = pd.DataFrame(y_pred)
    y_test_df['label'] = y_test_df['label'].map({0: 'healthy', 1: 'rubbish', 2: 'unhealthy'})

    y_test_df.to_csv(dest_filename, header=True, index=False)

    print(f"✅ Predictions saved to {dest_filename}")
