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
  - Autosens ratio vs p2u_relative plot (proves autosens blindness in closed loop)
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
ILLNESS_START_DAY      = SIM_DAYS +1 
ILLNESS_END_DAY        = SIM_DAYS +2 
CGM_SEED               = 1
TARGET_REDUCTION_FACTOR   = 1
TARGET_RAT_MULTIPLIER     = 1
MAX_GLUCOSE_OFFSET_MMOL_L = 0
WARMUP_DAYS            = 1
NO_ILLNESS = True

PATIENT_PARA_FILE = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")

def get_all_patients():
    return list(pd.read_csv(PATIENT_PARA_FILE)["Name"].values)

PATIENT_NAMES = get_all_patients()   # all 30 patients for weekend run
#PATIENT_NAMES = ["adult#009"]      # single patient for quick testing

SAVE_PATH = os.path.join(
    os.path.dirname(__file__),
    "examples", "results", "5Conditions_base_scenario",
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


def illness_factors(scenario, current_time):
    step_idx = scenario._step_index_5min(current_time)
    intensity = scenario._illness_intensity(step_idx)
    reduction_factor = 1.0 - (1.0 - scenario.target_reduction_factor) * intensity
    rat_multiplier   = 1.0 + (scenario.target_rat_multiplier  - 1.0) * intensity
    return reduction_factor, rat_multiplier


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

        base_p2u = float(env.patient._params.p2u)
        base_f   = float(env.patient._params.f)

        

        controller.reset()
        obs, reward, done, info = env.reset()

        while env.time < env.scenario.start_time + timedelta(days=SIM_DAYS):
            if not NO_ILLNESS:
                reduction_factor, rat_multiplier = illness_factors(scenario, env.time)
                env.patient._params.p2u = base_p2u * reduction_factor
                env.patient._params.f   = base_f   * rat_multiplier
            else: 
                reduction_factor, rat_multiplier = 1.0, 1.0
                env.patient._params.p2u = base_p2u
                env.patient._params.f   = base_f  

            # Collect illness trace data
            autotune_isf  = np.nan
            effective_isf = np.nan
            pump_isf      = np.nan
            autosens_ratio = np.nan
            sick_flag      = np.nan

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
            
            autosens_ratio = np.nan
            cfg = CONDITIONS[condition_key]
            if cfg.get("enable_autosens", False) and hasattr(controller, "manager") and patient_name in controller.manager.patients:
                autosens_ratio = controller.manager.patients[patient_name].autosens_ratio

            

            action = controller.policy(obs, reward, done, **info)
            obs, reward, done, info = env.step(action)

        env.patient._params.p2u = base_p2u
        env.patient._params.f   = base_f

        df_patient = env.show_history()
        df_patient.to_csv(os.path.join(save_path, f"{patient_name}.csv"))

        all_results.append(df_patient)
        histories[patient_name] = df_patient

       
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(PATIENT_NAMES)}] {patient_name} done in {elapsed:.0f}s")

    df_all = pd.concat(all_results, keys=PATIENT_NAMES)
    results, *_ = report(df_all, sensor_for_report, save_path)
    return results, histories

# ══════════════════════════════════════════════════════════════════════════════
#   METRICS AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def build_metrics_table(all_histories):
    """
    Build a per-patient metrics table across all conditions and time windows.
    all_histories: dict[condition_key → dict[patient_name → df]]
    """
    rows = []
    patients = sorted(set.intersection(*[set(h.keys()) for h in all_histories.values()]))

    if not NO_ILLNESS:
        windows = {
            "full":        (None,               None),
            "pre":         (None,               ILLNESS_START_TIME),
            "illness":     (ILLNESS_START_TIME,  ILLNESS_END_TIME),
            "post":        (ILLNESS_END_TIME,    None),
        }
    else: 
        windows = {"full": (None, None)}

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
#   PLOTS
# ══════════════════════════════════════════════════════════════════════════════


def save_bg_4conditions_plot(save_path, patient_name, all_histories):
    """5-panel BG trace, one per condition, illness window shaded."""
    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)

   
    cond_colors = {
        "A_Loop":     "tab:blue",
        "B_Autotune": "tab:red",
        "C_Autosens": "tab:orange",
        "D_Autotune_Autosens": "tab:green",
        "E_Full":     "tab:purple",
    }

    for ax, cond_key in zip(axes, CONDITIONS.keys()):
        df = all_histories.get(cond_key, {}).get(patient_name)
        color = cond_colors[cond_key]
        label = CONDITIONS[cond_key]["label"]

        if df is not None and "BG" in df.columns:
            ax.plot(df.index, df["BG"], color=color, linewidth=1.0, alpha=0.9)

        ax.axhline(70,  color="red",    linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(180, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
        
        ax.set_ylabel("BG (mg/dL)")
        ax.set_ylim(30, 350)
        ax.grid(True, alpha=0.2)

        # Compute and annotate TIR in illness window
    
        m_full = metrics_for_window(df, None, None)
        ax.set_title(
            f"{label} | TIR={m_full['TIR']:.1f}% "
            f"TAR={m_full['TAR']:.1f}% "
            f"TBR={m_full['TBR']:.1f}%"
        )

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{patient_name} | BG base scenario — 5 conditions", fontsize=13, fontweight="bold")
    fig.tight_layout()

    safe = patient_name.replace("#", "_")
    path = os.path.join(save_path, f"bg_5conditions_{safe}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_tir_comparison_plot(metrics_df, save_path):
    """
    Side-by-side plots: illness-window TIR and post-illness TBR for all conditions.
    Each patient is a line; conditions are x-axis ticks.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 15))

    cond_labels = [CONDITIONS[k]["label"] for k in CONDITIONS]
    x = list(range(len(cond_labels)))

    # Left panel: Illness-window TIR
    ax = axes[0]
    for patient in metrics_df.index:
        y = []
        for cond_key in CONDITIONS:
            col = f"{cond_key}_full_TIR"
            y.append(metrics_df.loc[patient, col] if col in metrics_df.columns else np.nan)
        if any(~np.isnan(y)):
            line, = ax.plot(x, y, marker="o", linewidth=1.2, alpha=0.75)
            ax.text(x[-1] + 0.05, y[-1], patient, fontsize=6,
                    va="center", color=line.get_color())

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=15, ha="right")
    ax.set_ylabel("Time In Range (%)")
    ax.set_title("TIR (Full 14 days)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    # Right panel: Post-illness TBR (key metric for autotune danger)
    ax = axes[1]
    for patient in metrics_df.index:
        y = []
        for cond_key in CONDITIONS:
            col = f"{cond_key}_full_TBR"
            y.append(metrics_df.loc[patient, col] if col in metrics_df.columns else np.nan)
        if any(~np.isnan(y)):
            line, = ax.plot(x, y, marker="o", linewidth=1.2, alpha=0.75)
            ax.text(x[-1] + 0.05, y[-1], patient, fontsize=6,
                    va="center", color=line.get_color())

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=15, ha="right")
    ax.set_ylabel("Time Below Range (%)")
    ax.set_title(
        "TBR (Full 14 days)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    fig.suptitle(
        "Ablation Study: Loop vs Autotune vs Autosens vs Full Adaptive\n"
        "Base scenario, 14-day simulation, 30 patients",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    path = os.path.join(save_path, "tir_tbr_5conditions.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_illness_profile_plot(save_path):
    scenario   = build_illness_scenario()
    total_steps = int((timedelta(days=SIM_DAYS).total_seconds() // 60) // 5)
    step_idx   = list(range(total_steps + 1))
    intensity  = [scenario._illness_intensity(s) for s in step_idx]
    reduction  = [1.0 - (1.0 - scenario.target_reduction_factor) * v for v in intensity]
    offset_mgdl = [scenario.max_glucose_offset_mgdl * v for v in intensity]
    day_axis   = [s / 288.0 for s in step_idx]

    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(day_axis, intensity,  label="Illness intensity",   linewidth=2)
    ax1.plot(day_axis, reduction,  label="Reduction factor",    linewidth=2)
    ax1.set_xlabel("Simulation day")
    ax1.set_ylabel("Unitless scale")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(day_axis, offset_mgdl, label="Glucose offset (mg/dL)", linewidth=2, linestyle="--")
    ax2.set_ylabel("Glucose offset (mg/dL)")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right", fontsize=9)
    fig.tight_layout()

    path = os.path.join(save_path, "illness_profile.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


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
        f"(p2u × {TARGET_REDUCTION_FACTOR}, glucose offset {MAX_GLUCOSE_OFFSET_MMOL_L} mmol/L)",
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
        for win in ["full"]:
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
            col = f"{cond_key}_full_TBR"
            val = metrics_df.loc[patient, col] if col in metrics_df.columns else np.nan
            row_str += f"  {val:>13.1f}%"
        lines.append(row_str)

    # TIR comparison
    lines += ["", "", "FULL TIR COMPARISON", "=" * 60, ""]
    lines.append(header)
    lines.append("-" * (20 + 16 * len(CONDITIONS)))
    for patient in metrics_df.index:
        row_str = f"{patient:<20}"
        for cond_key in CONDITIONS:
            col = f"{cond_key}_full_TIR"
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

    for cond_key in CONDITIONS:
        cond_save = os.path.join(SAVE_PATH, cond_key)
        report_results, histories = run_condition(cond_key, cond_save)
        all_report_results[cond_key] = report_results
        all_histories[cond_key]      = histories
        

    total_min = (time.time() - wall_start) / 60
    print(f"\nAll conditions done in {total_min:.1f} minutes")

    # Build metrics table
    print("\nBuilding metrics table...")
    metrics_df = build_metrics_table(all_histories)
    metrics_df.to_csv(os.path.join(SAVE_PATH, "metrics_all_conditions.csv"))

    # Plots
    print("\nGenerating plots...")
    #save_illness_profile_plot(SAVE_PATH)
    save_tir_comparison_plot(metrics_df, SAVE_PATH)

    # Per-patient plots 
    for patient_name in PATIENT_NAMES:
        save_bg_4conditions_plot(SAVE_PATH, patient_name, all_histories)

    # Summary
    write_summary(SAVE_PATH, metrics_df, all_report_results)

    # Print key results to terminal
    print("\n" + "=" * 80)
    print("TBR (mean across all patients)")
    print("=" * 80)
    for cond_key in CONDITIONS:
        col = f"{cond_key}_full_TBR"
        if col in metrics_df.columns:
            mean_tbr = metrics_df[col].mean()
            print(f"  {CONDITIONS[cond_key]['label']:30s}  post-illness TBR = {mean_tbr:.1f}%")

    print("\n" + "=" * 80)
    print("TIR (mean across all patients)")
    print("=" * 80)
    for cond_key in CONDITIONS:
        col = f"{cond_key}_full_TIR"
        if col in metrics_df.columns:
            mean_tir = metrics_df[col].mean()
            print(f"  {CONDITIONS[cond_key]['label']:30s}  illness TIR = {mean_tir:.1f}%")

    print(f"\nAll results saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()