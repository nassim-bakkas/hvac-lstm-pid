"""
End-to-end pipeline: load data -> train LSTM -> PID correction -> energy comparison.

Usage:
    python src/main.py --data data/temperature.csv
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from energy import REF_TEMP, calculate_energy, total_energy
from model import build_model, train
from pid import correct_predictions
from preprocessing import prepare


def plot_raw(temp, out_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(temp, label="Measured temperature")
    plt.title("Hourly temperature - measured data")
    plt.xlabel("Hours")
    plt.ylabel("Temperature (degC)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "01_raw_temperature.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


def plot_predictions(temp, train_pred, test_pred, look_back, q_real, q_pred, out_dir):
    train_plot = np.empty_like(temp, dtype=float)
    train_plot[:, :] = np.nan
    train_plot[look_back : look_back + len(train_pred)] = train_pred

    test_plot = np.empty_like(temp, dtype=float)
    test_plot[:, :] = np.nan
    start = look_back + len(train_pred)
    test_plot[start : start + len(test_pred)] = test_pred

    plt.figure(figsize=(12, 6))
    plt.plot(temp, label="Measured data", color="blue")
    plt.plot(train_plot, label="Predictions (train)", color="orange")
    plt.plot(test_plot, label="Predictions (test)", color="red")
    plt.title("Temperature predictions with LSTM")
    plt.xlabel("Hours")
    plt.ylabel("Temperature (degC)")
    plt.grid(True)
    plt.legend()
    plt.annotate(
        f"Q real total (24h)  : {q_real:.0f} J\nQ LSTM total (24h) : {q_pred:.0f} J",
        xy=(0.02, 0.95), xycoords="axes fraction", va="top",
        bbox=dict(boxstyle="round", fc="white", ec="black"),
    )
    plt.savefig(os.path.join(out_dir, "02_lstm_predictions.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


def plot_pid(temp, test_pred, corrected, look_back, n_train, out_dir):
    idx = np.arange(look_back + n_train, look_back + n_train + len(test_pred))

    plt.figure(figsize=(12, 6))
    plt.plot(temp, label="Measured data", color="royalblue")
    plt.plot(idx, test_pred, "r--", label="LSTM predictions (test)")
    plt.plot(idx, corrected, "g-", label="PID-corrected predictions")
    plt.title("LSTM predictions with and without PID correction")
    plt.xlabel("Hours")
    plt.ylabel("Temperature (degC)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "03_pid_correction.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/temperature.csv")
    p.add_argument("--figures", default="outputs",
                   help="where to write generated plots (figures/ holds the "
                        "published results from the original study)")
    p.add_argument("--look-back", type=int, default=48,
                   help="hours of context fed to the LSTM")
    p.add_argument("--test-size", type=int, default=24,
                   help="hours held out for testing")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-iqr", action="store_true",
                   help="skip IQR outlier filtering (recommended)")
    p.add_argument("--kp", type=float, default=0.5)
    p.add_argument("--ki", type=float, default=0.1)
    p.add_argument("--kd", type=float, default=0.05)
    args = p.parse_args()

    os.makedirs(args.figures, exist_ok=True)

    d = prepare(
        args.data,
        look_back=args.look_back,
        test_size=args.test_size,
        apply_iqr=not args.no_iqr,
    )
    scaler = d["scaler"]

    plot_raw(d["temp"], args.figures)

    model = build_model(look_back=args.look_back)
    train(
        model,
        d["x_train"],
        d["y_train"],
        validation_data=(d["x_test"], d["y_test"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    train_pred = scaler.inverse_transform(model.predict(d["x_train"]))
    test_pred = scaler.inverse_transform(model.predict(d["x_test"]))
    y_train_inv = scaler.inverse_transform(d["y_train"].reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(d["y_test"].reshape(-1, 1))

    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred))
    print(f"Train RMSE: {train_rmse:.2f} degC")
    print(f"Test  RMSE: {test_rmse:.2f} degC")

    # Energy over the 24 h test horizon
    q_real_arr, q_pred_arr = calculate_energy(
        y_test_inv.ravel(), test_pred.ravel(), ref_temp=REF_TEMP
    )
    q_real, q_pred = total_energy(q_real_arr), total_energy(q_pred_arr)
    delta = 100.0 * (q_real - q_pred) / q_real
    print(f"Q real total (24h) : {q_real:.0f} J")
    print(f"Q LSTM total (24h) : {q_pred:.0f} J")
    print(f"Difference          : {delta:.1f} %")

    plot_predictions(d["temp"], train_pred, test_pred, args.look_back,
                     q_real, q_pred, args.figures)

    corrected = correct_predictions(
        test_pred.ravel(), y_test_inv.ravel(),
        kp=args.kp, ki=args.ki, kd=args.kd,
    )
    plot_pid(d["temp"], test_pred.ravel(), corrected, args.look_back,
             len(train_pred), args.figures)

    corrected_rmse = np.sqrt(mean_squared_error(y_test_inv.ravel(), corrected))
    print(f"Test RMSE after PID correction: {corrected_rmse:.2f} degC")
    print(f"Figures written to {args.figures}/")


if __name__ == "__main__":
    main()
