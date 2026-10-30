"""Simulations for the validation of the midnight ISF adaptation (midnights.m)
of adaptiveController780G against a known change in the insulin sensitivity of
a simglucose patient. Plots and metrics: ISF_validation_plot.py.

    python AdaptiveMatlab/ISF_validation_simulate.py      # patients and cases from the SELECTION block below
    python AdaptiveMatlab/ISF_validation_simulate.py --patients adult#001 --cases illness   # override it
    python AdaptiveMatlab/ISF_validation_simulate.py --mode loop       # illness as in test_adaptive_vs_loop_base.py
    python AdaptiveMatlab/ISF_validation_simulate.py --controller-repo C:/path/AdaptiveControllerMatLab

Needs AdaptiveControllerMatLab branch IFB with midnights.m (commit 4054ca6
"ISF update at midnight" or later) AND the compact-state field isfLog
(AdaptiveMatlab/patches/0001-isfLog-compact-state.patch). Without isfLog the
run stops with a message. By default the controller is fetched from GitHub by
first_test.py; --controller-repo uses a local checkout instead. MATLAB is only
started when a 780G case is selected.

ILLNESS (the "true" sensitivity)
  Days 0..DAYS, illness from --illness-start to --illness-end [days], with the
  intensity profile of simglucose/simulation/illness_scenario.py (linear ramp
  up over ramp_fraction of the illness, plateau, linear ramp down). The true
  sensitivity factor is
      s(t) = 1 - (1 - reduction) * intensity(t)        (1 before / after)
  --mode sensitivity (default): s(t) multiplies Vmx and kp3 of the UVA/Padova
      patient (simglucose/patient/t1dpatient.py), i.e. both insulin effects:
      utilization  Vmt = Vm0 + Vmx * X  and production  EGP = kp1 - kp2*Gp - kp3*Id.
      This is the counterpart of ISF scaling kb1..kb3 in the Hovorka model.
  --mode loop: s(t) multiplies p2u and (1 + (rat-1) * intensity) multiplies f,
      as in test_adaptive_vs_loop_base.py. Note: p2u is the rate of insulin
      action, dX/dt = -p2u X + p2u (I - Ib); at steady state X = I - Ib, so p2u
      changes how fast insulin acts, not how much. s(t) is then not a
      steady-state sensitivity, and f adds glucose the controller does not know of.

CASES, same seed, meals and patient:
  illness           780G, illness, ISF adapted at midnight (midnights.m)
  no_illness        780G, no illness (s(t) = 1), ISF adapted; file suffix _no_illness
  illness_no_adapt  780G, illness, ISF NOT updated (cfg.AdaptationEnabled = false
                    through the bridge's adaptation=False; with midnights.m the
                    midnight call does nothing else); file suffix _illness_no_adapt
  illness_BB        simglucose BBController (target 140 mg/dL, CR/CF from
                    Quest.csv, meal bolus at each meal, no closed loop), illness,
                    no ISF; file suffix _illness_BB
  A case can be run on its own; files of other cases in --out-dir are kept.

OUTPUT (per patient and case, in --out-dir; file names carry the case suffix):
  <patient>.csv            simglucose history (BG, CGM, CHO, insulin, ...)
  <patient>_truth.csv      true factor s(t) and the patient parameters per step
  <patient>_isf.csv        controller ISF log, one row per midnight
                           [step, ISF_day, ISF, sum_phi2, n_samples]; empty without ISF update
  metadata.json            settings and controller commit (written by every run)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ================================================================ SELECTION
# Comment lines in or out to choose what is simulated. --patients / --cases on
# the command line override these lists.
PATIENTS = [
    #"child#001",
    #"adolescent#001",
    "adult#001",
]
CASES_TO_RUN = [
    "illness",              # 780G, illness, ISF adapted
    "no_illness",           # 780G, no illness, ISF adapted
    "illness_no_adapt",     # 780G, illness, ISF not updated
    "illness_BB",           # simglucose basal-bolus controller, illness
]
# ==========================================================================

SIMGLUCOSE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMGLUCOSE_ROOT))
OUTPUT_DIR = SIMGLUCOSE_ROOT / "simglucose_results" / "ISF_adaptation_validation" /"0.2_reduction_Umax_adapted_saved6daysestimate"

DAYS = 17
ILLNESS_START_DAY = 7
ILLNESS_END_DAY = 11
REDUCTION = 0.2                     # IllnessScenario default target_reduction_factor
RAT_MULTIPLIER = 1.3                # IllnessScenario default (mode loop only)
RAMP_FRACTION = 0.2                 # as test_adaptive_vs_loop_base.py
SEED = 1
SAMPLE_MIN = 5                      # GuardianRT
SPD = 24 * 60 // SAMPLE_MIN         # samples per day
MEALS = [(7, 45), (12, 70), (18, 80)]   # (hour, g) every day, as IFB_ablation.py
START = datetime(2020, 1, 1)        # midnight: the controller's clock starts at 00:00
ISF_COLUMNS = ["step", "ISF_day", "ISF", "sum_phi2", "n_samples", "Umax"]
CASES = {"illness": "", "no_illness": "_no_illness",          # case: file suffix
         "illness_no_adapt": "_illness_no_adapt", "illness_BB": "_illness_BB"}


# ---------------------------------------------------------------- illness
def build_scenario(days, start_day, end_day, reduction, rat):
    from simglucose.simulation.illness_scenario import IllnessScenario
    meals = [(24 * d + h, g) for d in range(days) for h, g in MEALS]
    return IllnessScenario(start_time=START, meal_schedule=meals,
                           illness_start_step=SPD * start_day,
                           illness_duration_steps=SPD * (end_day - start_day),
                           target_reduction_factor=reduction, target_rat_multiplier=rat,
                           max_glucose_offset_mmol_per_l=0.0, ramp_fraction=RAMP_FRACTION)


def true_factor(scenario, t, reduction):
    intensity = scenario._illness_intensity(scenario._step_index_5min(t))
    return 1.0 - (1.0 - reduction) * intensity, intensity


# ---------------------------------------------------------------- simulation
def controller_source(repo):
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}


def state_array(state, name):
    if name not in state:
        return None
    return np.atleast_2d(np.asarray(state[name], dtype=float))


def simulate(patients, cases, args, out_dir):
    from simglucose.actuator.pump import InsulinPump
    from simglucose.controller.basal_bolus_ctrller import BBController
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.simulation.env import T1DSimEnv

    need_matlab = any(c != "illness_BB" for c in cases)
    repo_url, source = None, {}
    if need_matlab:
        if args.controller_repo:
            repo = Path(args.controller_repo).resolve()
            repo_url = str(repo)
        else:
            import first_test                              # fetches branch IFB from GitHub
            repo, repo_url = first_test.BRIDGE_REPO, first_test.CONTROLLER_REPO_URL
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from simglucose_bridge import AdaptiveController, start_engine
        source = controller_source(repo)
        print(f"controller {repo_url} {source}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "metadata.json"
    old = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    meta = {
        "days": args.days, "illness_start_day": args.illness_start, "illness_end_day": args.illness_end,
        "reduction": args.reduction, "rat_multiplier": args.rat if args.mode == "loop" else 1.0,
        "ramp_fraction": RAMP_FRACTION, "mode": args.mode, "seed": args.seed, "meals": MEALS,
        "patients": sorted(set(old.get("patients", [])) | set(patients)),
        "cases": sorted(set(old.get("cases", [])) | set(cases)),
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }
    for k in ("days", "illness_start_day", "illness_end_day", "reduction", "mode", "seed"):
        if k in old and old[k] != meta[k]:
            raise ValueError(f"{mpath} was made with {k} = {old[k]}, this run has {meta[k]}; "
                             "use the same settings or another --out-dir")
    if need_matlab:
        meta.update({"controller_repo": repo_url, **{f"controller_{k}": v for k, v in source.items()}})
    else:                                                  # keep the controller info of earlier 780G runs
        meta.update({k: v for k, v in old.items() if k.startswith("controller_")})
    mpath.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    engine = start_engine(args.backend, str(repo)) if need_matlab else None
    try:
        for patient, case in [(p, c) for p in patients for c in cases]:
            ill = case != "no_illness"
            adapt = case in ("illness", "no_illness")        # 780G with midnight ISF update
            reduction = args.reduction if ill else 1.0
            rat = args.rat if (ill and args.mode == "loop") else 1.0
            name = patient + CASES[case]
            print(f"Running {patient} / {case} ({args.days} days, illness days {args.illness_start}-"
                  f"{args.illness_end}, plateau factor {reduction}, mode {args.mode})")
            scenario = build_scenario(args.days, args.illness_start, args.illness_end, reduction, rat)
            env = T1DSimEnv(T1DPatient.withName(patient), CGMSensor.withName("GuardianRT", seed=args.seed),
                            InsulinPump.withName("Insulet"), scenario)
            if case == "illness_BB":
                ctrl = BBController()                        # simglucose baseline
            else:
                ctrl = AdaptiveController(engine, controller="adaptive780G", adaptation=adapt)
            prm = env.patient._params
            base = {k: float(prm[k]) for k in ("Vmx", "kp3", "p2u", "f")}

            obs, reward, done, info = env.reset()
            truth = []
            while env.time < START + timedelta(days=args.days):
                s, intensity = true_factor(scenario, env.time, reduction)
                if args.mode == "sensitivity":
                    prm["Vmx"], prm["kp3"] = base["Vmx"] * s, base["kp3"] * s
                else:
                    prm["p2u"] = base["p2u"] * s
                    prm["f"] = base["f"] * (1.0 + (rat - 1.0) * intensity)
                truth.append((env.time, s, intensity, prm["Vmx"], prm["kp3"], prm["p2u"], prm["f"]))
                action = ctrl.policy(obs, reward, done, **info)
                obs, reward, done, info = env.step(action)
            for k, v in base.items():                       # leave the patient object as it was
                prm[k] = v

            env.show_history().to_csv(out_dir / f"{name}.csv")
            pd.DataFrame(truth, columns=["Time", "s_true", "intensity", "Vmx", "kp3", "p2u", "f"]
                         ).to_csv(out_dir / f"{name}_truth.csv", index=False)
            if adapt:
                isf = state_array(ctrl.controller_state(), "isfLog")
                if isf is None:
                    raise RuntimeError("The controller state has no 'isfLog'. Apply "
                                       "AdaptiveMatlab/patches/0001-isfLog-compact-state.patch to "
                                       "AdaptiveControllerMatLab (branch IFB) and run again.")
            else:
                isf = np.zeros((0, len(ISF_COLUMNS)))        # no midnight update: empty log
            pd.DataFrame(isf.reshape(-1, len(ISF_COLUMNS)), columns=ISF_COLUMNS
                         ).to_csv(out_dir / f"{name}_isf.csv", index=False)
    finally:
        if engine is not None:
            engine.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--patients", nargs="+", default=None,
                    help="simglucose patient names; default: PATIENTS in the SELECTION block")
    ap.add_argument("--cases", nargs="+", choices=list(CASES), default=None,
                    help="cases to simulate; default: CASES_TO_RUN in the SELECTION block")
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--illness-start", type=float, default=ILLNESS_START_DAY)
    ap.add_argument("--illness-end", type=float, default=ILLNESS_END_DAY)
    ap.add_argument("--reduction", type=float, default=REDUCTION, help="true factor at the illness plateau")
    ap.add_argument("--rat", type=float, default=RAT_MULTIPLIER, help="f multiplier at the plateau (mode loop)")
    ap.add_argument("--mode", choices=["sensitivity", "loop"], default="sensitivity")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--backend", choices=["matlab", "octave"], default="matlab")
    ap.add_argument("--controller-repo", default=None, help="local AdaptiveControllerMatLab checkout")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    patients = args.patients or PATIENTS
    cases = args.cases or CASES_TO_RUN
    if not patients or not cases:
        raise SystemExit("no patients or no cases selected (see the SELECTION block)")
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR / args.mode
    simulate(patients, cases, args, out_dir)
    print(f"done; plots: python AdaptiveMatlab/ISF_validation_plot.py --out-dir \"{out_dir}\"")


if __name__ == "__main__":
    main()