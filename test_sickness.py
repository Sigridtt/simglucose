"""
4-Condition ablation study: Loop vs Autotune-only vs Autosens-only vs Full Adaptive
Designed to run all 30 patients over a weekend.

Conditions:
  A — Loop only           (baseline, no adaptation)
  B — Loop + Autotune     (slow parameter update, no autosens, no sick detection)
  C — Loop + Autosens     (fast scaling, no autotune, no sick detection)
  D — Loop + Full Adaptive (autotune + autosens + sick detection)

Key outputs:
  - TIR/TAR/TBR per condition per window (pre-illness / illness / post-illness / full)
  - Autosens ratio vs Vmx_relative plot (proves autosens blindness in closed loop)
  - BG trace per condition for representative patients
  - Post-illness TBR (proves autotune danger without sick detection)
  - summary.txt with all metrics
"""

from datetime import datetime, timedelta
import os
import sys
import pandas as pd
import pkg_resources
import numpy as np
import matplotlib
import logging
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simglucose.simulation.illness_scenario import IllnessScenario
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.adaptive_loop_ctrller import AdaptiveLoopController
from simglucose.controller.loop_ctrller import LoopController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.analysis.report import report

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

sys.dont_write_bytecode = True

# ══════════════════════════════════════════════════════════════════════════════
#   CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

START_TIME             = datetime(2018, 1, 1, 8, 0, 0)
SIM_DAYS               = 14
ILLNESS_START_DAY      = 4
ILLNESS_END_DAY        = 10
CGM_SEED               = 1
TARGET_REDUCTION_FACTOR   = 0.2
TARGET_RAT_MULTIPLIER     = 1.60 # at peak illness, insulin needs are TARGET_REDUCTION_FACTOR% of baseline (Vmx × TARGET_REDUCTION_FACTOR) and glucose rises faster (rat × TARGET_RAT_MULTIPLIER)
MAX_GLUCOSE_OFFSET_MMOL_L = 0.0
WARMUP_DAYS            = 1
AUTOSENS_MAX             = 1.2  # Set a high max 1.2 in oref0

PATIENT_PARA_FILE = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")

def get_all_patients():
    return list(pd.read_csv(PATIENT_PARA_FILE)["Name"].values)

#PATIENT_NAMES = get_all_patients()   # all 30 patients for weekend run
PATIENT_NAMES = ["adult#003"] #, "child#007","child#008","adolescent#007" ]      # single patient for quick testing

SAVE_PATH = os.path.join(
    os.path.dirname(__file__),
    "examples", "results", "supersick_adult003_autosenspoints",
)

# Meal schedule
meal_schedule = []
for day in range(SIM_DAYS):
    offset = day * 24
    meal_schedule.append((offset + 1,  45))
    meal_schedule.append((offset + 5,  70))
    meal_schedule.append((offset + 11, 60))

# Time windows for sub-period metrics
ILLNESS_START_TIME  = START_TIME + timedelta(days=ILLNESS_START_DAY)
ILLNESS_END_TIME    = START_TIME + timedelta(days=ILLNESS_END_DAY)
SIM_END_TIME        = START_TIME + timedelta(days=SIM_DAYS)


# ══════════════════════════════════════════════════════════════════════════════
#   CONDITION DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

CONDITIONS = {
    
    "A_Loop": {
        "label": "A: Loop only",
        "use_adaptive": False,
    },
    
    "B_Autotune": {
        "label": "B: Loop + Autotune",
        "use_adaptive": True,
        "enable_autotune": True,
        "enable_autosens": False,
        "enable_sick_detection": False,
    },
    "C_Autosens": {
        "label": "C: Loop + Autosens",
        "use_adaptive": True,
        "enable_autotune": False,
        "enable_autosens": True,
        "enable_sick_detection": False,
    },
    "D_Autotune_Autosens": {
        "label": "D: Autotune + Autosens",
        "use_adaptive": True,
        "enable_autotune": True,
        "enable_autosens": True,
        "enable_sick_detection": False,
    },
    "E_Full": {
        "label": "E: Full Adaptive",
        "use_adaptive": True,
        "enable_autotune": True,
        "enable_autosens": True,
        "enable_sick_detection": True,
    },

}





# ══════════════════════════════════════════════════════════════════════════════
#   HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def disable_blocking_figures():
    plt.ioff()
    plt.show = lambda *a, **kw: None


def build_illness_scenario():
    return IllnessScenario(
        start_time=START_TIME,
        meal_schedule=meal_schedule,
        illness_start_step=288 * ILLNESS_START_DAY,
        illness_duration_steps=288 * (ILLNESS_END_DAY - ILLNESS_START_DAY),
        target_reduction_factor=TARGET_REDUCTION_FACTOR,
        target_rat_multiplier=TARGET_RAT_MULTIPLIER,
        max_glucose_offset_mmol_per_l=MAX_GLUCOSE_OFFSET_MMOL_L,
        ramp_fraction=0.2,
    )




def build_controller(condition_key):
    cfg = CONDITIONS[condition_key]
    if not cfg["use_adaptive"]:
        return LoopController(
            target=110,
            recommendation_type="automaticBolus",
            use_tdd_settings=True,
            insulin_type="novolog",
        )
    
    return AdaptiveLoopController(
        target=110,
        recommendation_type="automaticBolus",
        use_tdd_settings=True,
        insulin_type="novolog",
        warmup_days=WARMUP_DAYS,
        enable_autotune=cfg.get("enable_autotune", True),
        enable_autosens=cfg.get("enable_autosens", True),
        enable_sick_detection=cfg.get("enable_sick_detection", True),
        autosens_max= AUTOSENS_MAX,
    )


def metrics_for_window(df, t_start=None, t_end=None):
    """Compute TIR/TAR/TBR for a time window of a BG history dataframe."""
    if df is None or df.empty or "BG" not in df.columns:
        return {"TIR": np.nan, "TAR": np.nan, "TBR": np.nan}
    bg = df["BG"]
    if t_start is not None:
        bg = bg[bg.index >= t_start]
    if t_end is not None:
        bg = bg[bg.index < t_end]
    if bg.empty:
        return {"TIR": np.nan, "TAR": np.nan, "TBR": np.nan}
    return {
        "TIR": 100.0 * ((bg >= 70) & (bg <= 180)).mean(),
        "TAR": 100.0 * (bg > 180).mean(),
        "TBR": 100.0 * (bg < 70).mean(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#   SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_condition(condition_key, save_path):
    """Run one condition for all patients. Returns histories and illness traces."""
    os.makedirs(save_path, exist_ok=True)
    label = CONDITIONS[condition_key]["label"]
    print(f"\nRunning condition {label} ...")
    

    all_results    = []
    histories      = {}
    illness_traces = {}

    sensor_for_report = CGMSensor.withName("GuardianRT", seed=CGM_SEED)

    for i, patient_name in enumerate(PATIENT_NAMES):
        t0 = time.time()
        scenario = build_illness_scenario()
        patient  = T1DPatient.withName(patient_name)
        sensor   = CGMSensor.withName("GuardianRT", seed=CGM_SEED)
        pump     = InsulinPump.withName("Insulet")

        env        = T1DSimEnv(patient, sensor, pump, scenario)
        controller = build_controller(condition_key)

        base_Vmx = float(env.patient._params.Vmx)
        base_f   = float(env.patient._params.f)
        base_kp3 = float(env.patient._params.kp3) 

        illness_rows = []

        controller.reset()
        obs, reward, done, info = env.reset()

        while env.time < env.scenario.start_time + timedelta(days=SIM_DAYS):
            step_idx  = scenario._step_index_5min(env.time)
            intensity = scenario._illness_intensity(step_idx)

            vmx_factor = 1.0 - (1.0 - TARGET_REDUCTION_FACTOR)     * intensity
            kp3_factor = 1.0 - (1.0 - TARGET_REDUCTION_FACTOR) * intensity
            rat_multiplier = 1.0 + (TARGET_RAT_MULTIPLIER - 1.0)   * intensity

            env.patient._params.Vmx = base_Vmx * vmx_factor
            env.patient._params.kp3 = base_kp3 * kp3_factor
            env.patient._params.f   = base_f   * rat_multiplier

            # Collect illness trace data
            autotune_isf  = np.nan
            effective_isf = np.nan
            pump_isf      = np.nan
            autosens_ratio = np.nan
            sick_flag      = np.nan

            # saving autosens timestamps for visualization
            autosens_t_start = np.nan
            autosens_t_end   = np.nan

            if hasattr(controller, "get_current_isf"):
                try: autotune_isf = controller.get_current_isf(patient_name)
                except Exception: pass

            if hasattr(controller, "get_current_effective_isf"):
                try: effective_isf = controller.get_current_effective_isf(patient_name)
                except Exception: pass

            if hasattr(controller, "get_current_pump_isf"):
                try: pump_isf = controller.get_current_pump_isf(patient_name)
                except Exception: pass

            if hasattr(controller, "manager") and patient_name in controller.manager.patients:
                state = controller.manager.patients[patient_name]
                autosens_ratio = state.autosens_ratio
                sick_flag      = float(state.sick)
            
            
            cfg = CONDITIONS[condition_key]
            if cfg.get("enable_autosens", False) and hasattr(controller, "manager") and patient_name in controller.manager.patients:
                autosens_ratio = controller.manager.patients[patient_name].autosens_ratio
                autosens_t_start = state.autosens_window_t_start if state.autosens_window_t_start is not None else np.nan
                autosens_t_end   = state.autosens_window_t_end   if state.autosens_window_t_end   is not None else np.nan

            illness_rows.append({
                "Time":            env.time,
                "Vmx_relative":    env.patient._params.Vmx / base_Vmx if base_Vmx else np.nan,
                "kp3_relative":    env.patient._params.kp3 / base_kp3 if base_kp3 else np.nan,   
                "reduction_factor": vmx_factor,
                "rat_multiplier":  rat_multiplier,
                "autosens_ratio":  autosens_ratio,
                "autosens_t_start": autosens_t_start,
                "autosens_t_end":   autosens_t_end,
                "sick":            sick_flag,
                "autotune_isf":    autotune_isf,
                "effective_isf":   effective_isf,
                "pump_isf":        pump_isf,
                "bg":              obs.CGM if hasattr(obs, "CGM") else np.nan,
            })

            action = controller.policy(obs, reward, done, **info)
            obs, reward, done, info = env.step(action)

        env.patient._params.Vmx = base_Vmx
        env.patient._params.f   = base_f
        env.patient._params.kp3 = base_kp3 

        df_patient = env.show_history()
        df_patient.to_csv(os.path.join(save_path, f"{patient_name}.csv"))

        all_results.append(df_patient)
        histories[patient_name] = df_patient

        if illness_rows:
            illness_df = pd.DataFrame(illness_rows)
            illness_df["Time"] = pd.to_datetime(illness_df["Time"])
            illness_df = illness_df.set_index("Time")
            illness_df.to_csv(os.path.join(save_path, f"{patient_name}_illness_trace.csv"))
            illness_traces[patient_name] = illness_df

        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(PATIENT_NAMES)}] {patient_name} done in {elapsed:.0f}s")

    df_all = pd.concat(all_results, keys=PATIENT_NAMES)
    results, *_ = report(df_all, sensor_for_report, save_path)
    return results, histories, illness_traces


# ══════════════════════════════════════════════════════════════════════════════
#   METRICS AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def build_metrics_table(all_histories, all_report_results):
    """
    Build a per-patient metrics table across all conditions and time windows.
    all_histories: dict[condition_key → dict[patient_name → df]]
    all_report_results: dict[condition_key → dict[patient_name → df]]
    """
    rows = []
    patients = sorted(set.intersection(*[set(h.keys()) for h in all_histories.values()]))

    windows = {
        "full":        (None,               None),
        "pre":         (None,               ILLNESS_START_TIME),
        "illness":     (ILLNESS_START_TIME,  ILLNESS_END_TIME),
        "post":        (ILLNESS_END_TIME,    None),
    }

    for patient in patients:
        row = {"Patient": patient}
        for cond_key in CONDITIONS:
            df = all_histories.get(cond_key, {}).get(patient)
            for win_name, (t0, t1) in windows.items():
                m = metrics_for_window(df, t0, t1)
                for metric, val in m.items():
                    row[f"{cond_key}_{win_name}_{metric}"] = val
        rows.append(row)

    return pd.DataFrame(rows).set_index("Patient").sort_index()


# ══════════════════════════════════════════════════════════════════════════════
#   SUMMARY WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(save_path, metrics_df, condition_report_results):
    """Write comprehensive summary.txt with all metrics and condition comparisons."""
    lines = [
        "=" * 100,
        "ABLATION STUDY: Loop vs Autotune vs Autosens vs Full Adaptive",
        "=" * 100,
        f"Patients        : {len(PATIENT_NAMES)} ({', '.join(PATIENT_NAMES[:5])}...)",
        f"Simulation      : {SIM_DAYS} days",
        f"Illness window  : Day {ILLNESS_START_DAY}–{ILLNESS_END_DAY} "
        f"(Vmx × {TARGET_REDUCTION_FACTOR},  kp3 × {TARGET_REDUCTION_FACTOR}, "f"rat × {TARGET_RAT_MULTIPLIER})",
        f"Warmup          : {WARMUP_DAYS} day(s)",
        "",
        "CONDITIONS",
        "-" * 40,
    ]
    for k, v in CONDITIONS.items():
        lines.append(f"  {k:15s}  {v['label']}")
    lines.append("")

    # Per-condition aggregate means
    lines += ["", "AGGREGATE MEANS (across all patients)", "=" * 60]
    for cond_key in CONDITIONS:
        label = CONDITIONS[cond_key]["label"]
        lines.append(f"\n  {label}")
        for win in ["full", "pre", "illness", "post"]:
            cols_tir = f"{cond_key}_{win}_TIR"
            cols_tar = f"{cond_key}_{win}_TAR"
            cols_tbr = f"{cond_key}_{win}_TBR"
            if cols_tir in metrics_df.columns:
                tir = metrics_df[cols_tir].mean()
                tar = metrics_df[cols_tar].mean()
                tbr = metrics_df[cols_tbr].mean()
                lines.append(
                    f"    {win:10s}  TIR={tir:5.1f}%  TAR={tar:5.1f}%  TBR={tbr:5.1f}%"
                )

    # Post-illness TBR comparison (key danger metric)
    lines += ["", "", "POST-ILLNESS TBR COMPARISON (days 10–14)", "=" * 60,
              "B_Autotune ↑TBR vs A_Loop = autotune danger without sick detection",
              "E_Full ↓TBR vs B_Autotune = sick detection benefit",
              ""]

    header = f"{'Patient':<20}"
    for cond_key in CONDITIONS:
        header += f"  {cond_key:>14}"
    lines.append(header)
    lines.append("-" * (20 + 16 * len(CONDITIONS)))

    for patient in metrics_df.index:
        row_str = f"{patient:<20}"
        for cond_key in CONDITIONS:
            col = f"{cond_key}_post_TBR"
            val = metrics_df.loc[patient, col] if col in metrics_df.columns else np.nan
            row_str += f"  {val:>13.1f}%"
        lines.append(row_str)

    # Illness TIR comparison
    lines += ["", "", "ILLNESS-WINDOW TIR COMPARISON (days 4–10)", "=" * 60, ""]
    lines.append(header)
    lines.append("-" * (20 + 16 * len(CONDITIONS)))
    for patient in metrics_df.index:
        row_str = f"{patient:<20}"
        for cond_key in CONDITIONS:
            col = f"{cond_key}_illness_TIR"
            val = metrics_df.loc[patient, col] if col in metrics_df.columns else np.nan
            row_str += f"  {val:>13.1f}%"
        lines.append(row_str)



    # Full metrics table
    lines += ["", "", "FULL METRICS TABLE", "=" * 60, ""]
    lines.append(metrics_df.round(2).to_string())

    # Report results from simglucose report()
    lines += ["", "", "SIMGLUCOSE REPORT RESULTS", "=" * 60]
    for cond_key, report_df in condition_report_results.items():
        lines.append(f"\n  {CONDITIONS[cond_key]['label']}")
        if report_df is not None:
            lines.append(pd.DataFrame(report_df).round(3).to_string())

    path = os.path.join(save_path, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary written: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    disable_blocking_figures()
    os.makedirs(SAVE_PATH, exist_ok=True)

    wall_start = time.time()

    # Run all conditions
    all_report_results = {}
    all_histories      = {}
    all_illness_traces = {}

    for cond_key in CONDITIONS:
        cond_save = os.path.join(SAVE_PATH, cond_key)
        report_results, histories, illness_traces = run_condition(cond_key, cond_save)
        all_report_results[cond_key] = report_results
        all_histories[cond_key]      = histories
        all_illness_traces[cond_key] = illness_traces

    total_min = (time.time() - wall_start) / 60
    print(f"\nAll conditions done in {total_min:.1f} minutes")

    # Build metrics table
    print("\nBuilding metrics table...")
    metrics_df = build_metrics_table(all_histories, all_report_results)
    metrics_df.to_csv(os.path.join(SAVE_PATH, "metrics_all_conditions.csv"))

    # Summary
    write_summary(SAVE_PATH, metrics_df, all_report_results)

    # Print key results to terminal
    print("\n" + "=" * 80)
    print("KEY RESULTS — Post-illness TBR (mean across all patients)")
    print("=" * 80)
    for cond_key in CONDITIONS:
        col = f"{cond_key}_post_TBR"
        if col in metrics_df.columns:
            mean_tbr = metrics_df[col].mean()
            print(f"  {CONDITIONS[cond_key]['label']:30s}  post-illness TBR = {mean_tbr:.1f}%")

    print("\n" + "=" * 80)
    print("KEY RESULTS — Illness-window TIR (mean across all patients)")
    print("=" * 80)
    for cond_key in CONDITIONS:
        col = f"{cond_key}_illness_TIR"
        if col in metrics_df.columns:
            mean_tir = metrics_df[col].mean()
            print(f"  {CONDITIONS[cond_key]['label']:30s}  illness TIR = {mean_tir:.1f}%")

    print(f"\nAll results saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()