"""Illness test of adaptiveController780G for the simglucose test patients:
ISF adaptation with and without CGM filtering for the PID, and without ISF
adaptation. One BG plot per patient with all cases.

    python AdaptiveMatlab/SIMC_form_comparison.py      # what the SELECTION block says
    python AdaptiveMatlab/SIMC_form_comparison.py --controller-repo C:/path/AdaptiveControllerMatLab

Needs AdaptiveControllerMatLab branch IFB with midnights.m, simcFromModel.m and
Tune_SIMC_V2.m (one SIMC tuning: series PID with tauD = tau2, converted to the
parallel gains) and cfg.PIDFilterCGM. By default the controller is fetched from
GitHub by first_test.py; --controller-repo uses a local checkout.

CASES (same patient, seed, meals and illness; see CASE_SETTINGS):
  adapt            ISF adapted at midnight (midnights.m), Umax and PID gains from
                   the adapted model; PID on CGM filtered with FIlter_BGLV1
  adapt_no_filter  as adapt, but the PID uses the raw CGM (cfg.PIDFilterCGM = false)
  no_adapt         ISF NOT adapted (cfg.AdaptationEnabled = false through the
                   bridge's adaptation=False): midnights.m is not called, so ISF,
                   Umax and the PID gains stay at their values from init (step
                   test at ISF = 1); PID on filtered CGM

ILLNESS: from ILLNESS_START_DAY to ILLNESS_END_DAY with the intensity profile of
  simglucose/simulation/illness_scenario.py (ramp up, plateau, ramp down). The
  true sensitivity factor s(t) = 1 - (1 - REDUCTION) * intensity(t) multiplies
  Vmx and kp3 of the UVA/Padova patient (both insulin effects), as in
  ISF_validation_simulate.py (mode sensitivity).

OUTPUT (in OUT_DIR):
  <patient>_<case>.csv       simglucose history (BG, CGM, CHO, insulin, ...)
  <patient>_<case>_isf.csv   controller ISF log, one row per midnight (empty for no_adapt)
  <patient>_<case>_kp.csv    PID gains per update [step, Kp, Ki, Kd] (empty for no_adapt)
  <patient>_BG.png           BG of all cases, illness shaded
  summary.txt, metrics.csv   glucose metrics per period and case (Battelino et
                             al., Diabetes Care 2019: TIR 70-180, TBR <70/<54, TAR >180/>250)
  metadata.json              settings and controller commit
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================================================================ SELECTION
# Comment lines in or out.
PATIENTS = [
    "child#001",
    "adolescent#001",
    "adult#001",
]
CASES = [
    "adapt",            # ISF adapted, PID on filtered CGM
    "adapt_no_filter",  # ISF adapted, PID on raw CGM
    "no_adapt",         # ISF, Umax and PID gains stay at their values from init
]
# case: (ISF adaptation on, cfg overrides for the controller, plot colour, plot label)
CASE_SETTINGS = {
    "adapt":           (True,  {"PIDFilterCGM": True},  "tab:blue",   "780G, ISF adapted, PID on filtered CGM"),
    "adapt_no_filter": (True,  {"PIDFilterCGM": False}, "tab:orange", "780G, ISF adapted, PID on raw CGM"),
    "no_adapt":        (False, {"PIDFilterCGM": True},  "tab:purple", "780G, ISF not adapted"),
}
DO_SIMULATE = True                  # False: only make plots/summary from files in OUT_DIR
DO_PLOT = True
# ==========================================================================

SIMGLUCOSE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMGLUCOSE_ROOT))
OUT_DIR = SIMGLUCOSE_ROOT / "simglucose_results" / "PID_filter_comparison"

DAYS = 17
ILLNESS_START_DAY = 7
ILLNESS_END_DAY = 11
REDUCTION = 0.2                     # true sensitivity factor at the illness plateau
RAMP_FRACTION = 0.2                 # as test_adaptive_vs_loop_base.py
SEED = 1
SAMPLE_MIN = 5                      # GuardianRT
SPD = 24 * 60 // SAMPLE_MIN         # samples per day
MEALS = [(7, 45), (12, 70), (18, 80)]   # (hour, g) every day, as IFB_ablation.py
START = datetime(2020, 1, 1)        # midnight: the controller's clock starts at 00:00
CRASH_MGDL = 10.0
ISF_COLUMNS = ["step", "ISF_day", "ISF", "sum_phi2", "n_samples", "Umax"]
KP_COLUMNS = ["step", "Kp", "Ki", "Kd"]
STYLE = {"series": ("tab:blue", "780G, ISF adapted, SIMC series"),
         "parallel": ("tab:orange", "780G, ISF adapted, SIMC parallel"),
         "no_adapt": ("tab:purple", "780G, ISF not adapted")}


# ---------------------------------------------------------------- simulation
def build_scenario():
    from simglucose.simulation.illness_scenario import IllnessScenario
    meals = [(24 * d + h, g) for d in range(DAYS) for h, g in MEALS]
    return IllnessScenario(start_time=START, meal_schedule=meals,
                           illness_start_step=SPD * ILLNESS_START_DAY,
                           illness_duration_steps=SPD * (ILLNESS_END_DAY - ILLNESS_START_DAY),
                           target_reduction_factor=REDUCTION, target_rat_multiplier=1.0,
                           max_glucose_offset_mmol_per_l=0.0, ramp_fraction=RAMP_FRACTION)


def controller_source(repo):
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}


def log_table(state, name, columns):
    """Numeric log of the compact state as a DataFrame (empty if missing)."""
    if not state or name not in state:
        return pd.DataFrame(columns=columns)
    a = np.asarray(state[name], dtype=float)
    return pd.DataFrame(a.reshape(-1, len(columns)) if a.size else np.zeros((0, len(columns))), columns=columns)


def simulate(controller_repo, backend):
    if controller_repo:
        repo = Path(controller_repo).resolve()
        repo_url = str(repo)
    else:
        import first_test                                  # fetches branch IFB from GitHub
        repo, repo_url = first_test.BRIDGE_REPO, first_test.CONTROLLER_REPO_URL
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from simglucose.actuator.pump import InsulinPump
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.simulation.env import T1DSimEnv
    from simglucose_bridge import AdaptiveController, start_engine

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = controller_source(repo)
    print(f"controller {repo_url} {source}")
    (OUT_DIR / "metadata.json").write_text(json.dumps({
        "controller_repo": repo_url, **{f"controller_{k}": v for k, v in source.items()},
        "days": DAYS, "illness_start_day": ILLNESS_START_DAY, "illness_end_day": ILLNESS_END_DAY,
        "reduction": REDUCTION, "ramp_fraction": RAMP_FRACTION, "seed": SEED, "meals": MEALS,
        "patients": PATIENTS, "cases": CASES, "case_settings": {c: {"adaptation": a, "cfg_overrides": o} for c, (a, o, _, _) in CASE_SETTINGS.items()},
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

    engine = start_engine(backend, str(repo))
    try:
        for patient in PATIENTS:
            for case in CASES:
                print(f"Running {patient} / {case}")
                adapt, overrides, _, _ = CASE_SETTINGS[case]
                scenario = build_scenario()
                env = T1DSimEnv(T1DPatient.withName(patient), CGMSensor.withName("GuardianRT", seed=SEED),
                                InsulinPump.withName("Insulet"), scenario)
                ctrl = AdaptiveController(engine, controller="adaptive780G", adaptation=adapt,
                                          cfg_overrides=overrides)
                prm = env.patient._params
                base = {k: float(prm[k]) for k in ("Vmx", "kp3")}
                obs, reward, done, info = env.reset()
                while env.time < START + timedelta(days=DAYS):
                    intensity = scenario._illness_intensity(scenario._step_index_5min(env.time))
                    s = 1.0 - (1.0 - REDUCTION) * intensity
                    prm["Vmx"], prm["kp3"] = base["Vmx"] * s, base["kp3"] * s
                    action = ctrl.policy(obs, reward, done, **info)
                    obs, reward, done, info = env.step(action)
                for k, v in base.items():                   # leave the patient object as it was
                    prm[k] = v
                name = f"{patient}_{case}"
                env.show_history().to_csv(OUT_DIR / f"{name}.csv")
                state = ctrl.controller_state()
                log_table(state, "isfLog", ISF_COLUMNS).to_csv(OUT_DIR / f"{name}_isf.csv", index=False)
                log_table(state, "kpLog", KP_COLUMNS).to_csv(OUT_DIR / f"{name}_kp.csv", index=False)
    finally:
        engine.close()


# ---------------------------------------------------------------- analysis
def glucose_metrics(bg):
    bg = np.asarray(bg, dtype=float)
    bg = bg[np.isfinite(bg)]
    if not len(bg):
        return {}
    return {"mean": bg.mean(), "min": bg.min(), "max": bg.max(),
            "TIR70_180": 100 * np.mean((bg >= 70) & (bg <= 180)),
            "TBR70": 100 * np.mean(bg < 70), "TBR54": 100 * np.mean(bg < 54),
            "TAR180": 100 * np.mean(bg > 180), "TAR250": 100 * np.mean(bg > 250)}


def plot_and_summarize():
    meta = json.loads((OUT_DIR / "metadata.json").read_text(encoding="utf-8"))
    d0, d1, days = meta["illness_start_day"], meta["illness_end_day"], meta["days"]
    periods = {"pre": (0, d0), "illness": (d0, d1), "post": (d1, days), "all": (0, days)}
    rows = []
    for patient in PATIENTS:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axvspan(d0, d1, color="tab:red", alpha=0.08, label=f"illness (factor {meta['reduction']})")
        for case in CASES:
            f = OUT_DIR / f"{patient}_{case}.csv"
            if not f.exists():
                print(f"missing {f.name}, skipped")
                continue
            h = pd.read_csv(f, parse_dates=["Time"])
            day = (h["Time"] - pd.Timestamp(START)).dt.total_seconds() / 86400.0
            _, _, color, label = CASE_SETTINGS[case]
            ax.plot(day, h["BG"], lw=0.9, color=color, label=label)
            bg = h["BG"].to_numpy(float)[1:]                 # BG after each control step
            crash = np.flatnonzero(bg < CRASH_MGDL)
            n_valid = int(crash[0]) if len(crash) else len(bg)
            for name, (a, b) in periods.items():
                sl = slice(int(a * SPD), min(int(b * SPD), n_valid))
                rows.append({"patient": patient, "case": case, "period": name, "crashed": bool(len(crash)),
                             **glucose_metrics(bg[sl])})
        for y in (70, 180):
            ax.axhline(y, color="tab:green", ls="--", lw=0.8)
        ax.set_xlim(0, days)
        ax.set_xlabel("day")
        ax.set_ylabel("BG [mg/dL]")
        ax.set_title(f"{patient}: BG with illness, ISF adaptation with/without CGM filter vs no ISF adaptation")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{patient}_BG.png", dpi=150)
        plt.close(fig)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.float_format", "{:.1f}".format):
        table = metrics.to_string(index=False)
    text = (f"SIMC form comparison, illness days {d0}-{d1}, factor {meta['reduction']}, "
            f"controller {meta.get('controller_commit', '?')}\n\nBG metrics per period (mg/dL, %):\n{table}\n")
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controller-repo", default=None, help="local AdaptiveControllerMatLab checkout")
    ap.add_argument("--backend", choices=["matlab", "octave"], default="matlab")
    args = ap.parse_args()
    if DO_SIMULATE:
        simulate(args.controller_repo, args.backend)
    if DO_PLOT:
        plot_and_summarize()


if __name__ == "__main__":
    main()
