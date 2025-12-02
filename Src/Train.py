import argparse
import os
import numpy as np
from joblib import dump
from src.data import load_data
from src.model import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-out", default="models/model.joblib")
    args = parser.parse_args()

    X, y = load_data(args.data_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_model(X_train)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print("Validation RMSE:", rmse)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    dump(pipeline, args.model_out)
    print("Model saved to:", args.model_out)

if __name__ == "__main__":
    main()
