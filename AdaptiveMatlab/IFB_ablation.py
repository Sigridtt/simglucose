"""Ablation runs for the 780G-like controller: which change does what?

Each configuration changes one thing relative to adaptiveControllerV4 (no theta
adaptation), using the controller overrides of the simglucose bridge; the
adaptiveController780G preset must reproduce the combined configuration, and
simglucose's basal-bolus controller is the baseline.

    python AdaptiveMatlab/ablation_test.py
    python AdaptiveMatlab/ablation_test.py --configs V4 V4_IFB 780G --seeds 1 2 3
    python AdaptiveMatlab/ablation_test.py --reanalyze      # summary/plots from saved files only

Needs AdaptiveControllerMatLab branch IFB (8b647f4 or later) plus the bridge
controller overrides (ifb_enabled, meal_loop_max_factor, bolus_iob_risk_check,
auto_corr_horizon_min). The controller source and commit (and whether the
checkout had uncommitted changes) are written to metadata.json.

Metrics follow the International Consensus on Time in Range (Battelino et al.,
Diabetes Care 2019): TIR 70-180, TBR <70 / <54, TAR >180 / >250 mg/dL, CV.
A run is cut at the first BG < CRASH_MGDL and marked crashed; samples after it
are not physiological and are excluded.
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
OUTPUT_DIR = SIMGLUCOSE_ROOT / "simglucose_results" / "adaptive_matlab_IFB_ablation_v2"
PARAMS = SIMGLUCOSE_ROOT / "simglucose" / "params"

PATIENTS = ["child#001", "adolescent#001", "adult#001"]
DAYS = 10
SEEDS = [1]
SAMPLE_MIN = 5                      # GuardianRT
SPD = 24 * 60 // SAMPLE_MIN         # samples per day
MEALS = [(7, 45), (12, 70), (18, 80)]
CRASH_MGDL = 10.0                   # BG below this: simulation no longer physiological
PERIODS = {"days1-3": (0, 3), "days4-end": (3, None), "all": (0, None)}

NOCAPS = dict(meal_loop_max_factor=float("inf"), bolus_iob_risk_check=False)
# name: (controller, keyword arguments of AdaptiveController); None = simglucose BBController
CONFIGS = {
    "V4": ("adaptiveV4", {}),                                         # reference
    "V4_IFB": ("adaptiveV4", dict(ifb_enabled=True)),                 # + IFB only
    "V4_nocaps": ("adaptiveV4", NOCAPS),                              # + both pig caps off only
    "V4_nomealcap": ("adaptiveV4", dict(meal_loop_max_factor=float("inf"))),
    "V4_noIOBrisk": ("adaptiveV4", dict(bolus_iob_risk_check=False)),
    "V4_IFB_nocaps": ("adaptiveV4", dict(ifb_enabled=True, **NOCAPS)),  # = 780G preset
    "780G": ("adaptive780G", {}),                                     # must equal V4_IFB_nocaps
    "V4_AC240": ("adaptiveV4", dict(auto_corr_horizon_min=240)),      # child-crash check
    "780G_AC240": ("adaptive780G", dict(auto_corr_horizon_min=240)),
    "BB": None,
}
BOLUS_TYPES = {1: "meal", 2: "correction", 3: "autocorr"}


def daily_meals(meals, days):
    return [(24 * day + hour, grams) for day in range(days) for hour, grams in meals]


def patient_table():
    vp = pd.read_csv(PARAMS / "vpatient_params.csv").set_index("Name")
    q = pd.read_csv(PARAMS / "Quest.csv").set_index("Name")
    return vp, q


# ---------------------------------------------------------------- metrics
def glucose_metrics(bg):
    bg = np.asarray(bg, dtype=float)
    bg = bg[np.isfinite(bg)]
    if not len(bg):
        return {}
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


def column(state, name, n):
    """1-D float array of a per-step state field (NaN if missing), length n."""
    v = np.full(n, np.nan)
    if state and name in state:
        a = np.asarray(state[name], dtype=float).reshape(-1)
        m = min(len(a), n)
        v[:m] = a[:m]
    return v


def insulin_metrics(state, sl, n_days, cr, carbs):
    """Insulin split (U/day) and IFB / loop activity from the controller state."""
    if not state:
        return {}
    n = sl.stop
    loop = column(state, "loopU", n)[sl]
    bolus = column(state, "bolusU", n)[sl]
    btype = column(state, "bolusType", n)[sl]
    pid = column(state, "pidU", n)[sl]
    ifb = column(state, "ifbEstimate", n)[sl]
    out = {"loop_U_d": np.nansum(loop) / n_days,
           "loop_zero_pct": 100 * np.mean(loop < 1e-9),
           "ifb_on": bool(np.isfinite(ifb).any()),
           "ifb_reduced_pct": 100 * np.mean(loop < pid - 1e-9) if np.isfinite(pid).any() else np.nan}
    for code, name in BOLUS_TYPES.items():
        out[f"{name}_U_d"] = np.nansum(bolus[btype == code]) / n_days
    want = np.nansum(carbs[sl]) / cr
    out["meal_bolus_pct_of_CR"] = 100 * out["meal_U_d"] * n_days / want if want > 0 else np.nan
    return out


def summarize(result, state, patient, config, seed, patients_tab):
    vp, q = patients_tab
    bg_all = result["BG"].to_numpy(dtype=float)[1:]                  # BG after each control step
    ins_all = result["insulin"].to_numpy(dtype=float)[:-1] * SAMPLE_MIN   # U in that step
    carbs_all = result["CHO"].to_numpy(dtype=float)[:-1] * SAMPLE_MIN     # g eaten in that step
    crash = np.flatnonzero(bg_all < CRASH_MGDL)
    n_valid = int(crash[0]) if len(crash) else len(bg_all)
    rows = []
    for period, (d0, d1) in PERIODS.items():
        i0 = d0 * SPD
        i1 = n_valid if d1 is None else min(d1 * SPD, n_valid)
        if i1 - i0 < SPD // 2:                                        # less than half a day left
            continue
        sl = slice(i0, i1)
        n_days = (i1 - i0) / SPD
        rows.append({"patient": patient, "config": config, "seed": seed, "period": period,
                     "crashed": bool(len(crash)), "crash_hour": n_valid * SAMPLE_MIN / 60 if len(crash) else np.nan,
                     **glucose_metrics(bg_all[sl]),
                     "TDD": np.nansum(ins_all[sl]) / n_days,
                     "basalOnlyTDD": vp.loc[patient, "u2ss"] * vp.loc[patient, "BW"] / 6000.0 * 1440,
                     **insulin_metrics(state, sl, n_days, float(q.loc[patient, "CR"]), carbs_all)})
    adapt = np.asarray(state.get("adaptation", []), dtype=float).reshape(-1, 7) if state else np.zeros((0, 7))
    if len(adapt) and rows:
        rows[-1].update({"thetaEnd": adapt[-1, 2], "UmaxEnd": adapt[-1, 3], "KpEnd": adapt[-1, 4]})
    return rows


# ---------------------------------------------------------------- simulation
def flatten_state(state_json):
    """Per-step fields come back as [[x], [y], ...]; store them as [x, y, ...]."""
    out = {}
    for k, v in state_json.items():
        if isinstance(v, list) and v and all(isinstance(e, list) and len(e) == 1 for e in v):
            v = [e[0] for e in v]
        out[k] = v
    return out


def controller_source(repo):
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}


def run(configs, patients, days, seeds):
    import first_test                                        # selects / fetches the controller repo
    from simglucose.actuator.pump import InsulinPump
    from simglucose.controller.basal_bolus_ctrller import BBController
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.scenario import CustomScenario
    from simglucose.simulation.sim_engine import SimObj
    from simglucose_bridge import AdaptiveController, start_engine

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = controller_source(first_test.BRIDGE_REPO)
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps({
        "controller_repo": first_test.CONTROLLER_REPO_URL, **{f"controller_{k}": v for k, v in source.items()},
        "days": days, "seeds": seeds, "meals": MEALS, "crash_mgdl": CRASH_MGDL,
        "configs": {k: v for k, v in CONFIGS.items() if k in configs},
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2, default=str), encoding="utf-8")
    print(f"controller {source}")
    if source["uncommitted_changes"]:
        print("WARNING: controller checkout has uncommitted changes; results are not reproducible from the commit")

    meals = daily_meals(MEALS, days)
    engine = start_engine("matlab", str(first_test.BRIDGE_REPO))
    try:
        for seed in seeds:
            for config in configs:
                for patient in patients:
                    print(f"Running seed {seed} / {config} / {patient}")
                    env = T1DSimEnv(T1DPatient.withName(patient),
                                    CGMSensor.withName("GuardianRT", seed=seed),
                                    InsulinPump.withName("Insulet"),
                                    CustomScenario(start_time=datetime(2020, 1, 1), scenario=meals))
                    spec = CONFIGS[config]
                    if spec is None:
                        ctrl = BBController()
                    else:
                        controller, kwargs = spec
                        ctrl = AdaptiveController(engine, controller=controller, **kwargs)
                    sim = SimObj(env, ctrl, timedelta(days=days), animate=False)
                    sim.simulate()
                    out = OUTPUT_DIR / f"seed{seed}" / config
                    out.mkdir(parents=True, exist_ok=True)
                    sim.results().to_csv(out / f"{patient}.csv")          # keeps the Time index
                    if spec is not None:
                        state = flatten_state(first_test.state_to_jsonable(ctrl.controller_state()))
                        state["adaptationColumns"] = ["step", "nDays", "theta", "Umax", "Kp", "Ki", "Kd"]
                        (out / f"{patient}_controller_state.json").write_text(json.dumps(state), encoding="utf-8")
    finally:
        engine.close()


# ---------------------------------------------------------------- analysis
def load(seed, config, patient):
    out = OUTPUT_DIR / f"seed{seed}" / config
    csv = out / f"{patient}.csv"
    if not csv.exists():
        return None, None
    result = pd.read_csv(csv)
    sf = out / f"{patient}_controller_state.json"
    state = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else None
    return result, state


def check_preset(summary):
    """The 780G preset must reproduce V4_IFB_nocaps exactly (same seed / patient)."""
    a = summary[(summary.config == "780G") & (summary.period == "all")].set_index(["seed", "patient"])
    b = summary[(summary.config == "V4_IFB_nocaps") & (summary.period == "all")].set_index(["seed", "patient"])
    common = a.index.intersection(b.index)
    for key in common:
        same = np.isclose(a.loc[key, "mean"], b.loc[key, "mean"]) and np.isclose(a.loc[key, "TDD"], b.loc[key, "TDD"])
        print(f"780G == V4_IFB_nocaps for {key}: {same}")


def analyze(configs, patients, seeds):
    tab = patient_table()
    rows, daily = [], {}
    for seed in seeds:
        for config in configs:
            for patient in patients:
                result, state = load(seed, config, patient)
                if result is None:
                    continue
                rows += summarize(result, state, patient, config, seed, tab)
                bg = result["BG"].to_numpy(dtype=float)[1:]
                bg = np.where(np.arange(len(bg)) >= np.argmax(bg < CRASH_MGDL) if (bg < CRASH_MGDL).any() else False,
                              np.nan, bg)
                daily[(seed, config, patient)] = [np.nanmean(bg[d * SPD:(d + 1) * SPD]) for d in range(len(bg) // SPD)]
    if not rows:
        print(f"No results in {OUTPUT_DIR}")
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.precision", 1)
    cols = ["patient", "config", "seed", "period", "crashed", "mean", "TIR70_180", "TBR70", "TBR54",
            "TAR250", "TDD", "loop_U_d", "meal_U_d", "autocorr_U_d", "meal_bolus_pct_of_CR",
            "loop_zero_pct", "ifb_on", "ifb_reduced_pct"]
    print(summary[summary.period == "days4-end"][[c for c in cols if c in summary]].to_string(index=False))
    check_preset(summary)

    fig, axes = plt.subplots(1, len(patients), figsize=(6 * len(patients), 4.5), sharey=True, squeeze=False)
    for ax, patient in zip(axes[0], patients):
        for config in configs:
            curves = [daily[(s, config, patient)] for s in seeds if (s, config, patient) in daily]
            if curves:
                d = np.nanmean(np.array(curves, dtype=float), axis=0)
                ax.plot(np.arange(1, len(d) + 1), d, marker="o", ms=3, label=config,
                        linestyle="--" if config == "BB" else "-")
        ax.axhspan(70, 180, color="tab:green", alpha=0.08)
        ax.set_title(patient)
        ax.set_xlabel("Day")
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel(f"Daily mean BG (mg/dL), mean over seeds {seeds}")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "daily_mean_bg.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'summary.csv'} and {OUTPUT_DIR / 'daily_mean_bg.png'}")

    fig, axes = plt.subplots(1, len(patients), figsize=(6 * len(patients), 4.5),
                             sharey=True, squeeze=False)
    for ax, patient in zip(axes[0], patients):
        for config in configs:
            curves = []
            elapsed_hours = None
            for seed in seeds:
                result, _ = load(seed, config, patient)
                if result is None:
                    continue
                bg = result["BG"].to_numpy(dtype=float)
                crash = np.flatnonzero(bg[1:] < CRASH_MGDL)
                if len(crash):
                    bg[crash[0] + 1:] = np.nan
                if "Time" in result:
                    time = pd.to_datetime(result["Time"])
                    elapsed_hours = (time - time.iloc[0]).dt.total_seconds().to_numpy() / 3600
                else:
                    elapsed_hours = np.arange(len(result)) * SAMPLE_MIN / 60
                curves.append(bg)
            if curves:
                n = min(len(curve) for curve in curves)
                mean_bg = np.nanmean(np.array([curve[:n] for curve in curves]), axis=0)
                ax.plot(elapsed_hours[:n], mean_bg, linewidth=0.8, label=config)
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
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--reanalyze", action="store_true", help="skip simulation, analyse saved results")
    a = ap.parse_args(argv)
    if not a.reanalyze:
        run(a.configs, a.patients, a.days, a.seeds)
    analyze(a.configs, a.patients, a.seeds)


if __name__ == "__main__":
    main()