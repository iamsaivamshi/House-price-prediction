from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

def get_features(df: pd.DataFrame):
    num = df.select_dtypes(include=["number"]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return num, cat

def build_preprocessor(df):
    num, cat = get_features(df)

    num_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(sparse=False, handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_transformer, num),
        ("cat", cat_transformer, cat)
    ])

    return pre
