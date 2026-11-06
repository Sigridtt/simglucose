"""Plots and metrics for the ISF adaptation validation, from the files written by
ISF_validation_simulate.py. Needs no MATLAB and simulates nothing.

    python AdaptiveMatlab/ISF_validation_plot.py                # folder and patients from the SELECTION block
    python AdaptiveMatlab/ISF_validation_plot.py --patients adult#001
    python AdaptiveMatlab/ISF_validation_plot.py --out-dir C:/path/to/results   # overrides RESULT_FOLDER

Patients: PATIENTS in the SELECTION block; None = every patient that has files
in the folder. Cases that are missing for a patient are left out of its plot.

COMPARISON
  The controller's ISF is relative to its own (personalised Hovorka) model,
  not to the patient, so it is not 1 before the illness. It is therefore
  normalised by its value at the start of the illness (the ISF applied at the
  midnight that starts day illness_start, estimated from pre-illness data):
      ISF_norm = ISF / ISF(illness start)
  and compared with the true factor s(t). ISF_day (the estimate of one day,
  before the 6-day combination) is compared with the mean of s(t) over that
  same day; the applied ISF of a day was estimated from the days before it, so
  it lags.

PLOTS (per patient, <patient>_ISF.png)
  top     BG: illness with ISF adapted (black) and its CGM (blue), illness with
          ISF not updated (purple), illness with simglucose's BBController
          (brown), no illness (grey)
  middle  true factor s(t) (red) vs ISF_norm applied (blue step), ISF_day_norm
          (orange dots, at the middle of the day they describe), no-illness
          ISF normalised the same way (grey step)
  bottom  raw ISF applied and ISF_day, no-illness ISF, ISF not updated (= 1)
  ISF_all_patients.png: ISF_norm of all patients vs s(t)

OUTPUT (in --out-dir): the PNGs, <patient>[_no_illness]_isf_tracking.csv,
  metrics.csv and summary.txt (glucose and risk metrics per period and case,
  ISF tracking metrics). Metrics follow the International Consensus on Time in
  Range (Battelino et al., Diabetes Care 2019): TIR 70-180, TBR <70 / <54,
  TAR >180 / >250 mg/dL. Mean LBGI, HBGI, and Risk are also reported for
  each period, including pre-illness and post-illness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ISF_adaptation_validation import CASES, SAMPLE_MIN, SIMGLUCOSE_ROOT, SPD, START

# ================================================================ SELECTION
# Comment lines in or out. --out-dir / --patients on the command line override these.
RESULTS = SIMGLUCOSE_ROOT / "simglucose_results" / "ISF_adaptation_validation"
#RESULT_FOLDER = RESULTS / "0.2_reduction" / "sensitivity"
#RESULT_FOLDER = RESULTS / "0.2_reduction_Umax_adapted" / "sensitivity"
#RESULT_FOLDER = RESULTS / "0.2_reduction_Umax_adapted_saved6daysestimate" / "sensitivity"
#RESULT_FOLDER = RESULTS / "0.2_reduction_allAdaptations" / "sensitivity"
#RESULT_FOLDER = RESULTS / "0.2_reduction_allAdaptations" / "sensitivity" / "lambda_day0.6"
#RESULT_FOLDER = RESULTS / "0.8_reduction" / "sensitivity"
RESULT_FOLDER = RESULTS / "fuzzy_logic_bolusbasal" / "sensitivity" / "lambda_0.3"

# RESULT_FOLDER = RESULTS / "sensitivity"
# RESULT_FOLDER = RESULTS / "loop"

PATIENTS = None                      # None = all patients with files in RESULT_FOLDER
# PATIENTS = ["adult#001"]
# PATIENTS = ["child#001", "adolescent#001", "adult#001"]
# ==========================================================================

CRASH_MGDL = 10.0                   # BG below this: simulation no longer physiological


# ---------------------------------------------------------------- loading
def find_patients(out_dir):
    """Patient names that have files in out_dir (any case)."""
    names = set()
    for f in out_dir.glob("*_truth.csv"):
        name = f.name[:-len("_truth.csv")]
        for suffix in sorted((s for s in CASES.values() if s), key=len, reverse=True):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        names.add(name)
    return sorted(names)


def load(out_dir, patient):
    """patient may carry a case suffix; None if the run is missing."""
    if not (out_dir / f"{patient}_isf.csv").exists():
        return None
    hist = pd.read_csv(out_dir / f"{patient}.csv", parse_dates=["Time"])
    truth = pd.read_csv(out_dir / f"{patient}_truth.csv", parse_dates=["Time"])
    isf = pd.read_csv(out_dir / f"{patient}_isf.csv")
    days = lambda t: (t - pd.Timestamp(START)).dt.total_seconds() / 86400.0
    hist["day"], truth["day"] = days(hist["Time"]), days(truth["Time"])
    isf["day"] = (isf["step"] - 1) * SAMPLE_MIN / 1440.0       # midnight of the update
    return hist, truth, isf


# ---------------------------------------------------------------- metrics
def glucose_metrics(bg, metric_frame=None):
    bg = np.asarray(bg, dtype=float)
    bg = bg[np.isfinite(bg)]
    if not len(bg):
        return {}
    metrics = {"mean": bg.mean(), "min": bg.min(), "max": bg.max(),
            "TIR70_180": 100 * np.mean((bg >= 70) & (bg <= 180)),
            "TBR70": 100 * np.mean(bg < 70), "TBR54": 100 * np.mean(bg < 54),
            "TAR180": 100 * np.mean(bg > 180), "TAR250": 100 * np.mean(bg > 250)}
    if metric_frame is not None:
        for column in ("LBGI", "HBGI", "Risk"):
            if column in metric_frame:
                values = pd.to_numeric(metric_frame[column], errors="coerce").to_numpy()
                values = values[np.isfinite(values)]
                metrics[f"mean_{column}"] = float(values.mean()) if len(values) else np.nan
    return metrics


def period_rows(patient, case, hist, periods):
    """Glucose metrics per period; samples after the first BG < CRASH_MGDL are excluded."""
    samples = hist.iloc[1:].copy()                               # after each control step
    bg = samples["BG"].to_numpy(float)
    crash = np.flatnonzero(bg < CRASH_MGDL)
    n_valid = int(crash[0]) if len(crash) else len(bg)
    rows = []
    for name, (a, b) in periods.items():
        sl = slice(int(a * SPD), min(int(b * SPD), n_valid))
        rows.append({"patient": patient, "case": case, "period": name, "crashed": bool(len(crash)),
                     **glucose_metrics(bg[sl], samples.iloc[sl])})
    return rows


def isf_tracking(truth, isf, meta):
    """Normalised ISF vs the true factor, per midnight."""
    d0 = meta["illness_start_day"]
    ref_rows = isf[isf["day"] <= d0 + 1e-9]
    ref = float(ref_rows["ISF"].iloc[-1]) if len(ref_rows) else np.nan
    t = isf.copy()
    t["ISF_norm"] = t["ISF"] / ref
    t["ISF_day_norm"] = t["ISF_day"] / ref
    s_day = truth.groupby(np.floor(truth["day"] + 1e-9))["s_true"].mean()   # mean s per day
    t["s_true_prev_day"] = [s_day.get(round(d) - 1, np.nan) for d in t["day"]]   # day ISF_day describes
    t["s_true_next_day"] = [s_day.get(round(d), np.nan) for d in t["day"]]       # day the ISF is applied in
    return t, ref


def lag_days(t, level, after_day, falling):
    """First midnight after after_day where ISF_norm crosses level; minus after_day."""
    rows = t[t["day"] > after_day + 1e-9]
    hit = rows[(rows["ISF_norm"] <= level) if falling else (rows["ISF_norm"] >= level)]
    return float(hit["day"].iloc[0] - after_day) if len(hit) else np.nan


# ---------------------------------------------------------------- plots
def plot_patient(patient, hist, truth, track, ref, meta, out_dir, base=None, fixed=None, bb=None):
    """base: (hist, track, ref) of the no-illness run, drawn in grey; fixed / bb: history of
    the illness run without ISF update (purple) / with simglucose's BBController (brown);
    None = not available."""
    d0, d1 = meta["illness_start_day"], meta["illness_end_day"]
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                           gridspec_kw={"height_ratios": [1.3, 1, 0.8]})
    for a in ax:
        a.axvspan(d0, d1, color="tab:red", alpha=0.08, label="illness")
        a.grid(alpha=0.3)
    if base is not None:
        ax[0].plot(base[0]["day"], base[0]["BG"], lw=0.8, color="grey", alpha=0.8, label="BG, no illness")
    if bb is not None:
        ax[0].plot(bb["day"], bb["BG"], lw=0.8, color="tab:brown", alpha=0.8,
                   label="BG, illness, simglucose BB controller")
    if fixed is not None:
        ax[0].plot(fixed["day"], fixed["BG"], lw=0.8, color="tab:purple", alpha=0.8,
                   label="BG, illness, ISF not updated")
    ax[0].plot(hist["day"], hist["BG"], lw=0.8, color="black", label="BG, illness, ISF adapted")
    ax[0].plot(hist["day"], hist["CGM"], lw=0.6, color="tab:blue", alpha=0.6, label="CGM, illness")
    for y in (70, 180):
        ax[0].axhline(y, color="tab:green", ls="--", lw=0.8)
    ax[0].set_ylabel("glucose [mg/dL]")
    ax[0].set_title(f"{patient}: BG, true and adapted insulin sensitivity (mode {meta['mode']})")
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].plot(truth["day"], truth["s_true"], color="tab:red", lw=2, label="true factor s(t)")
    if base is not None:
        bt = base[1]
        ax[1].step(np.r_[bt["day"], meta["days"]], np.r_[bt["ISF_norm"], bt["ISF_norm"].iloc[-1]],
                   where="post", color="grey", lw=2, label="no illness: ISF / its ISF(illness start)")
    ax[1].step(np.r_[track["day"], meta["days"]], np.r_[track["ISF_norm"], track["ISF_norm"].iloc[-1]],
               where="post", color="tab:blue", lw=2, label="adapted ISF / ISF(illness start), applied")
    ax[1].plot(track["day"] - 0.5, track["ISF_day_norm"], "o", color="tab:orange", ms=5,
               label="ISF_day / ISF(illness start), at the day it describes")
    ax[1].set_ylabel("relative sensitivity [-]")
    ax[1].legend(loc="lower left", fontsize=8)

    ax[2].step(np.r_[0, track["day"], meta["days"]], np.r_[1, track["ISF"], track["ISF"].iloc[-1]],
               where="post", color="tab:blue", lw=2, label="ISF applied (raw)")
    ax[2].plot(track["day"] - 0.5, track["ISF_day"], "o", color="tab:orange", ms=5, label="ISF_day (raw)")
    if base is not None:
        bt = base[1]
        ax[2].step(np.r_[0, bt["day"], meta["days"]], np.r_[1, bt["ISF"], bt["ISF"].iloc[-1]],
                   where="post", color="grey", lw=2, label="no illness: ISF applied (raw)")
    if fixed is not None:
        ax[2].axhline(1.0, color="tab:purple", lw=1.5, label="ISF not updated (= 1)")
    ax[2].axhline(ref, color="grey", ls=":", lw=1, label=f"ISF at illness start = {ref:.3f}")
    ax[2].set_ylabel("ISF [-]")
    ax[2].set_xlabel("day")
    ax[2].legend(loc="lower left", fontsize=8)
    ax[2].set_xlim(0, meta["days"])
    fig.tight_layout()
    fig.savefig(out_dir / f"{patient}_ISF.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- analysis
def analyze(patients, out_dir):
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    d0, d1, red = meta["illness_start_day"], meta["illness_end_day"], meta["reduction"]
    periods = {"pre": (0, d0), "illness": (d0, d1), "post": (d1, meta["days"]), "all": (0, meta["days"])}
    rows, lines = [], []
    fig, axs = plt.subplots(1, 1, figsize=(12, 4.5))
    axs.axvspan(d0, d1, color="tab:red", alpha=0.08)
    plotted_truth = False
    for patient in patients:
        run = load(out_dir, patient)
        brun = load(out_dir, patient + CASES["no_illness"])
        base = None
        if brun is not None:
            btrack, bref = isf_tracking(brun[1], brun[2], meta)
            btrack.to_csv(out_dir / f"{patient}_no_illness_isf_tracking.csv", index=False)
            base = (brun[0], btrack, bref)
            rows += period_rows(patient, "no_illness", brun[0], periods)
            lines += [f"{patient} (no illness): ISF at day {d0:g} {bref:.3f}, at the last midnight "
                      f"{btrack['ISF'].iloc[-1]:.3f} (ISF_norm {btrack['ISF_norm'].iloc[-1]:.3f}, "
                      f"min {btrack['ISF_norm'].min():.3f}, max {btrack['ISF_norm'].max():.3f})"]
        frun = load(out_dir, patient + CASES["illness_no_adapt"])
        fixed = None
        if frun is not None:
            fixed = frun[0]
            rows += period_rows(patient, "illness_no_adapt", fixed, periods)
        bbrun = load(out_dir, patient + CASES["illness_BB"])
        bb = None
        if bbrun is not None:
            bb = bbrun[0]
            rows += period_rows(patient, "illness_BB", bb, periods)
        if run is None:
            print(f"no illness results for {patient}")
            continue
        hist, truth, isf = run
        track, ref = isf_tracking(truth, isf, meta)
        track.to_csv(out_dir / f"{patient}_isf_tracking.csv", index=False)
        plot_patient(patient, hist, truth, track, ref, meta, out_dir, base, fixed, bb)
        if not plotted_truth:
            axs.plot(truth["day"], truth["s_true"], color="black", lw=2, label="true factor s(t)")
            plotted_truth = True
        axs.step(np.r_[track["day"], meta["days"]], np.r_[track["ISF_norm"], track["ISF_norm"].iloc[-1]],
                 where="post", lw=1.5, label=patient)

        rows += period_rows(patient, "illness", hist, periods)
        ill = track[(track["day"] > d0 + 1e-9) & (track["day"] <= d1 + 1)]
        valid = track.dropna(subset=["s_true_prev_day"])
        valid = valid[valid["day"] >= 2]                             # skip the first nights (start-up)
        lines += [
            f"{patient}: ISF at illness start {ref:.3f}",
            f"  lowest ISF_norm during/after illness {ill['ISF_norm'].min():.3f} (true plateau {red:.2f})",
            f"  ISF_norm at the last midnight {track['ISF_norm'].iloc[-1]:.3f} (true 1.00)",
            f"  days from illness start until ISF_norm <= {(1 + red) / 2:.2f} (half the change): "
            f"{lag_days(track, (1 + red) / 2, d0, True):.0f}",
            f"  days from illness end until ISF_norm >= {(1 + red) / 2:.2f}: "
            f"{lag_days(track, (1 + red) / 2, d1, False):.0f}",
            f"  ISF_day_norm vs mean s of the same day (days 2-{meta['days']}): RMSE "
            f"{np.sqrt(np.mean((valid['ISF_day_norm'] - valid['s_true_prev_day']) ** 2)):.3f}, "
            f"mean error {np.mean(valid['ISF_day_norm'] - valid['s_true_prev_day']):+.3f}",
        ]
    axs.set_xlabel("day")
    axs.set_ylabel("ISF / ISF(illness start)")
    axs.set_title(f"Adapted vs true insulin sensitivity (mode {meta['mode']})")
    axs.grid(alpha=0.3)
    axs.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ISF_all_patients.png", dpi=150)
    plt.close(fig)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.float_format", "{:.1f}".format):
        table = metrics.to_string(index=False)
    text = (f"ISF adaptation validation, mode {meta['mode']}, illness days {d0}-{d1}, "
            f"reduction {red}, controller {meta.get('controller_commit', '?')}\n\n"
            f"BG metrics per period (mg/dL, %):\n{table}\n\nISF tracking:\n" + "\n".join(lines) + "\n")
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--patients", nargs="+", default=None,
                    help="simglucose patient names; default: PATIENTS in the SELECTION block")
    ap.add_argument("--out-dir", default=None, help="result folder; default: RESULT_FOLDER")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(RESULT_FOLDER)
    if not (out_dir / "metadata.json").exists():
        raise SystemExit(f"no metadata.json in {out_dir}; check RESULT_FOLDER / --out-dir")
    print(f"plotting from {out_dir}")
    patients = args.patients or PATIENTS or find_patients(out_dir)
    if not patients:
        raise SystemExit(f"no results in {out_dir}; run ISF_validation_simulate.py first")
    analyze(patients, out_dir)


if __name__ == "__main__":
    main()