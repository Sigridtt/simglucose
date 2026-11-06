"""Compare runs of ISF_adaptation_validation.py with different ISFLambdaDay.
Reads <base-dir>/lambda_<value>/ and makes one figure per patient:
  top     BG during illness, ISF adapted, for each lambda (+ ISF not adapted if available)
  bottom  true sensitivity factor s(t) and the applied ISF normalised by its value
          at the start of the illness, for each lambda
Also prints TIR / TBR / min BG per period and the ISF tracking per lambda.

    python AdaptiveMatlab/ISF_lambda_compare_plot.py --base-dir <OUTPUT_DIR>/sensitivity
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

START = datetime(2020, 1, 1)
SAMPLE_MIN = 5
SPD = 24 * 60 // SAMPLE_MIN
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def days(t):
    return (t - pd.Timestamp(START)).dt.total_seconds() / 86400.0


def metrics(bg):
    bg = bg[np.isfinite(bg)]
    return {"TIR70_180": 100 * np.mean((bg >= 70) & (bg <= 180)), "TBR70": 100 * np.mean(bg < 70),
            "TBR54": 100 * np.mean(bg < 54), "TAR180": 100 * np.mean(bg > 180), "min": bg.min()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-dir", required=True, help="folder that holds lambda_<value>/ subfolders")
    args = ap.parse_args()
    base = Path(args.base_dir)
    runs = sorted(base.glob("lambda_*"), key=lambda p: float(p.name.split("_", 1)[1]))
    if not runs:
        raise SystemExit(f"no lambda_* folders in {base}")
    meta = json.loads((runs[0] / "metadata.json").read_text(encoding="utf-8"))
    d0, d1, ndays = meta["illness_start_day"], meta["illness_end_day"], meta["days"]
    periods = {"pre": (0, d0), "illness": (d0, d1), "post": (d1, ndays)}
    patients = sorted({f.name[:-len("_isf.csv")] for f in runs[0].glob("*#*_isf.csv")
                       if f.name.count("_") == 1})
    rows = []
    for patient in patients:
        fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
        for a in ax:
            a.axvspan(d0, d1, color="tab:red", alpha=0.08)
            a.grid(alpha=0.3)
        truth_done = False
        for c, run in zip(COLORS, runs):
            lam = run.name.split("_", 1)[1]
            if not (run / f"{patient}.csv").exists():
                continue
            h = pd.read_csv(run / f"{patient}.csv", parse_dates=["Time"])
            ax[0].plot(days(h["Time"]), h["BG"], lw=0.8, color=c, label=f"BG, ISF adapted, lambda {lam}")
            bg = h["BG"].to_numpy(float)[1:]
            for name, (a, b) in periods.items():
                rows.append({"patient": patient, "lambda": lam, "period": name,
                             **metrics(bg[int(a * SPD):int(b * SPD)])})
            isf = pd.read_csv(run / f"{patient}_isf.csv")
            isf["day"] = (isf["step"] - 1) * SAMPLE_MIN / 1440.0
            ref_rows = isf[isf["day"] <= d0 + 1e-9]
            ref = float(ref_rows["ISF"].iloc[-1]) if len(ref_rows) else np.nan
            ax[1].step(np.r_[isf["day"], ndays], np.r_[isf["ISF"], isf["ISF"].iloc[-1]] / ref,
                       where="post", color=c, lw=2, label=f"applied ISF / ISF(illness start), lambda {lam}")
            ax[1].plot(isf["day"] - 0.5, isf["ISF_day"] / ref, "o", color=c, ms=4)
            if not truth_done:
                tr = pd.read_csv(run / f"{patient}_truth.csv", parse_dates=["Time"])
                ax[1].plot(days(tr["Time"]), tr["s_true"], color="black", lw=2, label="true factor s(t)")
                truth_done = True
        noad = runs[0] / f"{patient}_illness_no_adapt.csv"
        if noad.exists():
            h = pd.read_csv(noad, parse_dates=["Time"])
            ax[0].plot(days(h["Time"]), h["BG"], lw=0.8, color="tab:purple", alpha=0.7, label="BG, ISF not adapted")
        for y in (70, 180):
            ax[0].axhline(y, color="tab:green", ls="--", lw=0.8)
        ax[0].set_ylabel("BG [mg/dL]")
        ax[0].set_title(f"{patient}: ISFLambdaDay comparison (dots: ISF_day of each lambda)")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[1].set_ylabel("relative sensitivity [-]")
        ax[1].set_xlabel("day")
        ax[1].set_xlim(0, ndays)
        ax[1].legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(base / f"{patient}_lambda_compare.png", dpi=150)
        plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(base / "lambda_compare_metrics.csv", index=False)
    with pd.option_context("display.width", 140, "display.float_format", "{:.1f}".format):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()