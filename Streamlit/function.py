# function.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Expected input columns (order matters)
COLUMN_ORDER = [
    "Orbital Period (days)",
    "Transit Duration (hrs)",
    "Transit Depth (ppm)",
    "Planet Radius (Earth Radii)",
    "Planet Insolation Flux (Earth Flux)",
    "Planet Equilibrium Temperature (Kelvin)",
    "Star Effective Temperature (Kelvin)",
    "Star Radius (solar radii)",
]

def load_scaler_from_train(train_csv_path: str, drop_first_col: bool = True) -> StandardScaler:
    """
    Fits a StandardScaler on the training data. Matches your earlier approach
    using train.csv and dropping the first column.
    """
    x = np.loadtxt(train_csv_path, delimiter=",", skiprows=1)
    if drop_first_col:
        x = x[:, 1:]
    scaler = StandardScaler().fit(x)
    return scaler

def load_model(model_path: str):
    # Use tf.keras consistently
    return tf.keras.models.load_model(model_path)

def coerce_and_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has the required columns in the right order and coercible to float.
    If headers are missing but the column count matches, assign COLUMN_ORDER.
    """
    if not set(COLUMN_ORDER).issubset(df.columns) and df.shape[1] == len(COLUMN_ORDER):
        df = df.copy()
        df.columns = COLUMN_ORDER

    # Now validate presence
    missing = [c for c in COLUMN_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Reorder and coerce to float
    X = df[COLUMN_ORDER].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Non-numeric or NA values in columns: {bad}")
    return X

def predict_probs(df: pd.DataFrame, scaler: StandardScaler, model) -> np.ndarray:
    """
    Returns a 1D array of predicted probabilities.
    """
    X = coerce_and_order_columns(df)
    Xs = scaler.transform(X.values)
    # Suppress verbose spam in Streamlit reruns
    probs = model.predict(Xs, verbose=0).ravel()
    return probs