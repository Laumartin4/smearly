import pandas as pd
import numpy as np

def prediction_to_csv(y_pred: np.array, filename: str="predictions.csv"):
    """
    Save predictions to CSV file
    """
    y_pred = y_pred.argmax(axis=1)

    y_test_df = pd.read_csv("../raw_data/isbi2025-ps3c-test-dataset.csv")
    y_test_df["label"] = pd.DataFrame(y_pred)
    y_test_df['label'] = y_test_df['label'].map({0: 'healthy', 1: 'rubbish', 2: 'unhealthy'})

    y_test_df.to_csv(filename, header=True, index=False)

    print(f"✅ Predictions saved to {filename}")
