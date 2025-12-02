import pandas as pd

TARGET = "SalePrice"

def load_data(path):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
