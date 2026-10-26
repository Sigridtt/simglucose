"""Ablation runs for adaptiveControllerV3: which adaptation causes the drift?

Runs the same 14-day scenario for every patient with one adaptation setting
changed per configuration, plus simglucose's basal-bolus controller as a
baseline, and writes a weekly summary so the configurations can be compared.

    python AdaptiveMatlab/ablation_test.py
    python AdaptiveMatlab/ablation_test.py --configs full A_no_adaptation --days 7
    python AdaptiveMatlab/ablation_test.py --reanalyze      # summary/plots from saved files only

The controller is fetched from the AdaptiveControllerMatLab "test" branch by
first_test.py; its commit is written to metadata.json. The switches used
below (adapt_apply_model / adapt_apply_cf / adapt_apply_pid_gain /
adapt_limit) need commit 38a060d or later of that repository.

Metrics follow the International Consensus on Time in Range (Battelino et al.,
Diabetes Care 2019): TIR 70-180, TBR <70 / <54, TAR >180 / >250 mg/dL, CV.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIMGLUCOSE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = SIMGLUCOSE_ROOT / "simglucose_results" / "adaptive_matlab_IFB_ablation"
PARAMS = SIMGLUCOSE_ROOT / "simglucose" / "params"

PATIENTS = ["child#001", "adolescent#001", "adult#001"]
DAYS = 10
SEED = 1
SAMPLE_MIN = 5                      # GuardianRT
MEALS = [(7, 45), (12, 70), (18, 80)]

# One change per configuration (keyword arguments of AdaptiveControllerV3).
# "BB" is simglucose's basal-bolus controller (no MATLAB).
NO_APPLY = dict(adapt_apply_model=False, adapt_apply_cf=False, adapt_apply_pid_gain=False)
CONFIGS = {
    "full": {},                                                   # current default
    "A_no_adaptation": dict(adaptation=False),                    # no midnight adaptation at all
    "B_fit_only": NO_APPLY,                                       # theta fitted and logged, applied nowhere
    "C_pid_gain_only": {**NO_APPLY, "adapt_apply_pid_gain": True},
    "D_cf_only": {**NO_APPLY, "adapt_apply_cf": True},
    "E_model_only": {**NO_APPLY, "adapt_apply_model": True},
    "F_fixed_umax": dict(adapt_limit=False),                      # Umax fixed at 1.5 x basal
    "BB": None,
}


def daily_meals(meals, days):
    return [(24 * day + hour, grams) for day in range(days) for hour, grams in meals]


def patient_basal_u_per_sample(patient):
    vp = pd.read_csv(PARAMS / "vpatient_params.csv").set_index("Name")
    return float(vp.loc[patient, "u2ss"] * vp.loc[patient, "BW"] / 6000.0 * SAMPLE_MIN)


# ---------------------------------------------------------------- metrics
def glucose_metrics(bg):
    bg = np.asarray(bg, dtype=float)
    bg = bg[np.isfinite(bg)]
    return {
        "mean": bg.mean(),
        "CV": 100 * bg.std() / bg.mean(),
        "TIR70_180": 100 * np.mean((bg >= 70) & (bg <= 180)),
        "TBR70": 100 * np.mean(bg < 70),
        "TBR54": 100 * np.mean(bg < 54),
        "TAR180": 100 * np.mean(bg > 180),
        "TAR250": 100 * np.mean(bg > 250),
        "min": bg.min(),
        "max": bg.max(),
    }


def summarize(result, state, patient, config):
    """One row per week of the run, plus the adaptation state at the end."""
    spd = 24 * 60 // SAMPLE_MIN
    bg = result["BG"].to_numpy()[1:]
    ins = result["insulin"].to_numpy() * SAMPLE_MIN            # U per sample
    basal = patient_basal_u_per_sample(patient)
    adapt = np.asarray(state.get("adaptation", []), dtype=float).reshape(-1, 7) if state else np.zeros((0, 7))
    rows = []
    n_weeks = int(np.ceil(len(bg) / (7 * spd)))
    for w in range(n_weeks):
        sl = slice(w * 7 * spd, (w + 1) * 7 * spd)
        n_days = len(bg[sl]) / spd
        row = {"patient": patient, "config": config, "week": w + 1,
               **glucose_metrics(bg[sl]),
               "TDD": np.nansum(ins[sl]) / n_days,
               "basalOnlyTDD": basal * spd,
               "zeroInsulin": 100 * np.mean(ins[sl] == 0)}
        rows.append(row)
    if len(adapt):
        rows[-1].update({"thetaEnd": adapt[-1, 2], "UmaxEnd": adapt[-1, 3],
                         "KpStart": adapt[0, 4], "KpEnd": adapt[-1, 4], "KiEnd": adapt[-1, 5]})
    return rows


# ---------------------------------------------------------------- simulation
def run(configs, patients, days, seed):
    import first_test                                        # fetches the controller repo
    from simglucose.actuator.pump import InsulinPump
    from simglucose.controller.basal_bolus_ctrller import BBController
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.scenario import CustomScenario
    from simglucose.simulation.sim_engine import SimObj
    from simglucose_bridge import AdaptiveController780G, start_engine

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(first_test.BRIDGE_REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps({
        "controller_repo": first_test.CONTROLLER_REPO_URL,
        "controller_branch": first_test.CONTROLLER_REPO_BRANCH,
        "controller_commit": commit, "days": days, "seed": seed, "meals": MEALS,
        "configs": {k: v for k, v in CONFIGS.items() if k in configs},
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2, default=str), encoding="utf-8")
    print(f"controller commit {commit}")

    meals = daily_meals(MEALS, days)
    engine = start_engine("matlab", str(first_test.BRIDGE_REPO))
    try:
        for config in configs:
            for patient in patients:
                print(f"Running {config} / {patient}")
                env = T1DSimEnv(T1DPatient.withName(patient),
                                CGMSensor.withName("GuardianRT", seed=seed),
                                InsulinPump.withName("Insulet"),
                                CustomScenario(start_time=datetime(2020, 1, 1), scenario=meals))
                kwargs = CONFIGS[config]
                ctrl = BBController() if kwargs is None else AdaptiveController780G(engine, **kwargs)
                sim = SimObj(env, ctrl, timedelta(days=days), animate=False)
                sim.simulate()
                out = OUTPUT_DIR / config
                out.mkdir(exist_ok=True)
                sim.results().to_csv(out / f"{patient}.csv")          # keeps the Time index
                if kwargs is not None:
                    state = first_test.state_to_jsonable(ctrl.controller_state())
                    state["adaptationColumns"] = ["step", "nDays", "theta", "Umax", "Kp", "Ki", "Kd"]
                    (out / f"{patient}_controller_state.json").write_text(json.dumps(state), encoding="utf-8")
    finally:
        engine.close()


# ---------------------------------------------------------------- analysis
def load(config, patient):
    out = OUTPUT_DIR / config
    csv = out / f"{patient}.csv"
    if not csv.exists():
        return None, None
    result = pd.read_csv(csv)
    sf = out / f"{patient}_controller_state.json"
    state = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else None
    return result, state


def analyze(configs, patients):
    rows, daily = [], {}
    spd = 24 * 60 // SAMPLE_MIN
    for config in configs:
        for patient in patients:
            result, state = load(config, patient)
            if result is None:
                continue
            rows += summarize(result, state, patient, config)
            bg = result["BG"].to_numpy()[1:]
            daily[(config, patient)] = [np.mean(bg[d * spd:(d + 1) * spd]) for d in range(len(bg) // spd)]
    if not rows:
        print(f"No results in {OUTPUT_DIR}")
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 1)
    cols = ["patient", "config", "week", "mean", "TIR70_180", "TBR70", "TBR54", "TAR250",
            "TDD", "basalOnlyTDD", "zeroInsulin", "thetaEnd", "KpEnd"]
    print(summary[[c for c in cols if c in summary]].to_string(index=False))

    fig, axes = plt.subplots(1, len(patients), figsize=(6 * len(patients), 4.5), sharey=True, squeeze=False)
    for ax, patient in zip(axes[0], patients):
        for config in configs:
            if (config, patient) in daily:
                d = daily[(config, patient)]
                ax.plot(np.arange(1, len(d) + 1), d, marker="o", ms=3, label=config,
                        linestyle="--" if config == "BB" else "-")
        ax.axhspan(70, 180, color="tab:green", alpha=0.08)
        ax.set_title(patient)
        ax.set_xlabel("Day")
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("Daily mean BG (mg/dL)")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "daily_mean_bg.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'summary.csv'} and {OUTPUT_DIR / 'daily_mean_bg.png'}")

    fig, axes = plt.subplots(1, len(patients), figsize=(6 * len(patients), 4.5),
                             sharey=True, squeeze=False)
    for ax, patient in zip(axes[0], patients):
        for config in configs:
            result, _ = load(config, patient)
            if result is None:
                continue
            if "Time" in result:
                time = pd.to_datetime(result["Time"])
                elapsed_hours = (time - time.iloc[0]).dt.total_seconds() / 3600
            else:
                elapsed_hours = np.arange(len(result)) * SAMPLE_MIN / 60
            ax.plot(elapsed_hours, result["BG"], linewidth=0.8, label=config)
        ax.axhspan(70, 180, color="tab:green", alpha=0.08)
        ax.axhline(70, color="black", linewidth=0.6, alpha=0.5)
        ax.axhline(180, color="black", linewidth=0.6, alpha=0.5)
        ax.set_title(patient)
        ax.set_xlabel("Simulation time (hours)")
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("Blood glucose (mg/dL)")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bgl_over_time.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'bgl_over_time.png'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--patients", nargs="+", default=PATIENTS)
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--reanalyze", action="store_true", help="skip simulation, analyse saved results")
    a = ap.parse_args(argv)
    if not a.reanalyze:
        run(a.configs, a.patients, a.days, a.seed)
    analyze(a.configs, a.patients)


if __name__ == "__main__":
    main()