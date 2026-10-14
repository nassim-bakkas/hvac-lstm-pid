"""
Train and evaluate with a proper chronological train / validation / test split.

This is the honest counterpart to `main.py`, which reproduces the original
7-day run with its methodological flaws intact. Here:

  * validation and test are separate blocks, so early stopping never sees test;
  * the scaler is fitted on training data only;
  * the test set is a month rather than a single day.

Usage:
    python src/train_split.py --data data/temperature_2026.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy import REF_TEMP, calculate_energy, total_energy
from model import build_model, train
from pid import correct_predictions
from preprocessing import prepare_split


def persistence_baseline(y_true):
    """Naive forecast: tomorrow's hour equals this hour. The number any model
    has to beat before it has earned its complexity."""
    return np.concatenate([[y_true[0]], y_true[:-1]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/temperature_2026.csv")
    p.add_argument("--figures", default="outputs")
    p.add_argument("--look-back", type=int, default=48)
    p.add_argument("--val-size", type=int, default=720, help="hours")
    p.add_argument("--test-size", type=int, default=720, help="hours")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    import tensorflow as tf
    tf.random.set_seed(args.seed)

    os.makedirs(args.figures, exist_ok=True)

    d = prepare_split(
        args.data,
        look_back=args.look_back,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    scaler = d["scaler"]

    model = build_model(look_back=args.look_back)
    history = train(
        model,
        d["x_train"], d["y_train"],
        validation_data=(d["x_val"], d["y_val"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    def invert(a):
        return scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()

    test_pred = invert(model.predict(d["x_test"], verbose=0))
    y_test = invert(d["y_test"])

    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    mae = mean_absolute_error(y_test, test_pred)

    base = persistence_baseline(y_test)
    base_rmse = np.sqrt(mean_squared_error(y_test, base))

    corrected = np.asarray(correct_predictions(test_pred, y_test))
    corr_rmse = np.sqrt(mean_squared_error(y_test, corrected))

    # Gain sweep. Over a 24 h window the integral term is harmless; over a
    # month it accumulates, so this isolates integral windup as the cause of
    # any degradation rather than leaving it as a guess.
    sweep = []
    for kp, ki, kd in [(0.5, 0.1, 0.05), (0.5, 0.0, 0.05), (0.5, 0.0, 0.0),
                       (0.2, 0.0, 0.0), (0.5, 0.01, 0.05)]:
        c = np.asarray(correct_predictions(test_pred, y_test, kp=kp, ki=ki, kd=kd))
        sweep.append((kp, ki, kd, np.sqrt(mean_squared_error(y_test, c))))

    np.savetxt(
        os.path.join(args.figures, "test_predictions.csv"),
        np.column_stack([y_test, test_pred, corrected]),
        delimiter=",", header="measured,lstm,pid_corrected", comments="",
        fmt="%.4f",
    )

    q_real_arr, q_pred_arr = calculate_energy(y_test, test_pred, ref_temp=REF_TEMP)
    q_real, q_pred = total_energy(q_real_arr), total_energy(q_pred_arr)
    delta = 100.0 * (q_real - q_pred) / q_real

    print()
    print(f"test window           : {len(y_test)} hours")
    print(f"LSTM   test RMSE      : {rmse:.3f} degC")
    print(f"LSTM   test MAE       : {mae:.3f} degC")
    print(f"persistence baseline  : {base_rmse:.3f} degC")
    print(f"after PID correction  : {corr_rmse:.3f} degC")
    print(f"Q real  (test window) : {q_real:.0f} J")
    print(f"Q LSTM  (test window) : {q_pred:.0f} J")
    print(f"apparent 'saving'     : {delta:.2f} %")
    print()
    print("PID gain sweep (test RMSE, degC):")
    print(f"  {'Kp':>5} {'Ki':>6} {'Kd':>6}   RMSE")
    for kp, ki, kd, r in sweep:
        print(f"  {kp:5.2f} {ki:6.3f} {kd:6.3f}   {r:.3f}")

    # loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.title("Model loss (held-out validation set)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (scaled)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.figures, "04_loss_curve.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # test-window predictions
    plt.figure(figsize=(13, 5))
    plt.plot(y_test, label="Measured", color="royalblue", linewidth=1.2)
    plt.plot(test_pred, label="LSTM prediction", color="crimson",
             linewidth=1.2, alpha=0.85)
    plt.axhline(REF_TEMP, color="grey", linestyle=":", label="setpoint 22 °C")
    plt.title(f"Held-out test window ({len(y_test)} h) — RMSE {rmse:.2f} °C")
    plt.xlabel("Hours")
    plt.ylabel("Temperature (°C)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(args.figures, "05_test_window.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    print(f"\nFigures written to {args.figures}/")


if __name__ == "__main__":
    main()
