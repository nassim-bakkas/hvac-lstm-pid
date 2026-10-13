"""
Data loading and preparation for the LSTM.

Mirrors the preprocessing used in the original TIPE study:
missing-value interpolation, IQR outlier filtering, min-max scaling,
and sliding-window dataset construction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_series(path, column="temperature"):
    """Load a temperature series from CSV or Excel."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        data = pd.read_excel(path)
    else:
        data = pd.read_csv(path)

    if data[column].isnull().sum() > 0:
        print("Missing values detected. Filling by interpolation.")
        data[column] = data[column].interpolate()

    return data


def filter_outliers(data, column="temperature"):
    """Drop points outside 1.5 x IQR.

    NOTE: this is kept for fidelity with the original study, but it is a
    questionable choice for a time series -- it removes genuine temperature
    extremes and breaks the hourly continuity the LSTM relies on. See the
    Limitations section of the README. Prefer `filter_outliers=False` unless
    you are specifically reproducing the original run.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    before = len(data)
    data = data[(data[column] >= lower) & (data[column] <= upper)]
    print(f"IQR filter removed {before - len(data)} point(s).")
    return data


def scale(values):
    """Scale to [0, 1]. Returns (scaled_values, fitted_scaler)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(values), scaler


def create_dataset(series, look_back=48):
    """Build sliding windows: X[i] = series[i : i+look_back], y[i] = series[i+look_back]."""
    x, y = [], []
    for i in range(len(series) - look_back - 1):
        x.append(series[i : (i + look_back), 0])
        y.append(series[i + look_back, 0])
    return np.array(x), np.array(y)


def prepare(path, look_back=48, test_size=24, apply_iqr=True, column="temperature"):
    """Full pipeline: load -> (filter) -> scale -> window -> train/test split."""
    data = load_series(path, column)
    if apply_iqr:
        data = filter_outliers(data, column)

    temp = data[column].values.reshape(-1, 1)
    temp_scaled, scaler = scale(temp)

    x, y = create_dataset(temp_scaled, look_back)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    train_size = len(x) - test_size
    return {
        "x_train": x[:train_size],
        "x_test": x[train_size:],
        "y_train": y[:train_size],
        "y_test": y[train_size:],
        "scaler": scaler,
        "temp": temp,
        "look_back": look_back,
    }
