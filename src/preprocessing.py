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
    """Original pipeline: load -> (filter) -> scale -> window -> train/test split.

    Kept so the published 7-day results can be reproduced. It has two known
    flaws, both documented in the README: there is no validation set separate
    from the test set, and the scaler is fitted on the whole series, which lets
    the test range leak into training. Use `prepare_split` instead for an
    honest evaluation.
    """
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


def _window(series, look_back):
    x, y = create_dataset(series, look_back)
    return np.reshape(x, (x.shape[0], x.shape[1], 1)), y


def prepare_split(path, look_back=48, test_size=720, val_size=720,
                  apply_iqr=False, column="temperature"):
    """Chronological train / validation / test pipeline.

    Three differences from `prepare`, each removing a specific flaw:

    1. The series is cut into three *contiguous* blocks before windowing, so no
       training window ever overlaps validation or test data.
    2. The scaler is fitted on the training block alone. Fitting it on the whole
       series would tell the model the min and max of data it has not seen.
    3. Validation and test are separate, so early stopping never reads the test
       set.

    `test_size` and `val_size` are counted in hours. Defaults of 720 give one
    month each, which needs a dataset of at least a few months.
    """
    data = load_series(path, column)
    if apply_iqr:
        data = filter_outliers(data, column)

    temp = data[column].values.reshape(-1, 1)
    n = len(temp)
    if n < look_back + val_size + test_size + 48:
        raise ValueError(
            f"series has {n} points, too short for look_back={look_back}, "
            f"val_size={val_size}, test_size={test_size}"
        )

    cut_test = n - test_size
    cut_val = cut_test - val_size

    train_raw = temp[:cut_val]
    # Each block carries `look_back` hours of preceding context so its first
    # window is complete, without that context ever becoming a target.
    val_raw = temp[cut_val - look_back : cut_test]
    test_raw = temp[cut_test - look_back :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)

    x_train, y_train = _window(scaler.transform(train_raw), look_back)
    x_val, y_val = _window(scaler.transform(val_raw), look_back)
    x_test, y_test = _window(scaler.transform(test_raw), look_back)

    print(f"train {len(x_train)} | val {len(x_val)} | test {len(x_test)} windows")

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "scaler": scaler,
        "temp": temp,
        "look_back": look_back,
        "test_start": cut_test,
    }
