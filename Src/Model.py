from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from joblib import dump, load
from .features import build_preprocessor
import pandas as pd

def build_model(df: pd.DataFrame):
    preprocessor = build_preprocessor(df)
    model = RandomForestRegressor(n_estimators=150, random_state=42)

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])

    return pipeline

def save_model(model, path):
    dump(model, path)

def load_model(path):
    return load(path)
