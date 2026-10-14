"""
Diagnose whether anything is left in the LSTM's forecast residuals.

A causal correction (see `correct_predictions_causal` in pid.py) can only help
if successive forecast errors are correlated. This script measures that
directly from the saved test predictions, so the claim is a measurement rather
than an assertion.

Usage:
    python src/analyze_residuals.py --predictions outputs/test_predictions.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pid import residual_autocorrelation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="outputs/test_predictions.csv")
    p.add_argument("--out", default="outputs")
    p.add_argument("--max-lag", type=int, default=24)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = pd.read_csv(args.predictions)
    residuals = (d["measured"] - d["lstm"]).values
    n = len(residuals)

    acf = residual_autocorrelation(residuals, max_lag=args.max_lag)
    lags = sorted(acf)
    vals = [acf[l] for l in lags]

    # 95 % confidence band for white noise
    band = 1.96 / np.sqrt(n)

    print(f"residuals: n={n}  mean={residuals.mean():+.4f}  "
          f"sd={residuals.std():.4f}")
    print(f"white-noise 95% band: +/-{band:.4f}")
    inside = sum(abs(v) < band for v in vals)
    print(f"lags within the band: {inside}/{len(vals)}")
    for l in lags[:6]:
        print(f"  lag {l:2d}: {acf[l]:+.4f}")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhspan(-band, band, color="grey", alpha=0.18,
               label="95 % white-noise band")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.vlines(lags, 0, vals, color="#2563eb", linewidth=2)
    ax.plot(lags, vals, "o", color="#1d4ed8", markersize=5)
    ax.set_title("Autocorrelation of LSTM forecast residuals "
                 f"({n} h held-out test window)")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    lim = max(0.15, 1.25 * max(abs(v) for v in vals))
    ax.set_ylim(-lim, lim)
    peak = max(lags, key=lambda l: abs(acf[l]))
    ax.annotate(f"lag {peak}: {acf[peak]:+.2f}\ndaily structure the LSTM missed",
                xy=(peak, acf[peak]), xytext=(peak - 8, acf[peak] * 0.72),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#c2410c", lw=1.4),
                color="#c2410c")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    path = os.path.join(args.out, "08_residual_acf.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
