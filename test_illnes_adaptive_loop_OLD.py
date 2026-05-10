from datetime import datetime, timedelta
import os
import sys
import pandas as pd
import pkg_resources
import numpy as np
import matplotlib
import logging

from simglucose.simulation import env

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simglucose.simulation.illness_scenario import IllnessScenario
from simglucose.simulation.env import T1DSimEnv, Observation
from simglucose.controller.adaptive_loop_ctrller import AdaptiveLoopController
from simglucose.controller.loop_ctrller import LoopController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.analysis.report import report
from simglucose.analysis.risk import risk_index
logger = logging.getLogger(__name__)

sys.dont_write_bytecode = True

START_TIME = datetime(2018, 1, 1, 8, 0, 0)
SIM_DAYS = 14
ILLNESS_START_DAY = 4
ILLNESS_END_DAY = 10
CGM_SEED = 1
TARGET_REDUCTION_FACTOR = 0.20
TARGET_RAT_MULTIPLIER = 1.30
MAX_GLUCOSE_OFFSET_MMOL_PER_L = 2.0

PATIENT_PARA_FILE = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")


def get_all_patients():
    patient_df = pd.read_csv(PATIENT_PARA_FILE)
    return list(patient_df["Name"].values)


# Run all patients by default.
PATIENT_NAMES = get_all_patients()
#PATIENT_NAMES = ["adult#009"]
# Build meal schedule the same way as in test_compare_7day_all_patients
SAVE_PATH = os.path.join(
    os.path.dirname(__file__),
    "examples",
    "results",
    "supersick",
)

meal_schedule = []
for day in range(SIM_DAYS):
    offset = day * 24
    meal_schedule.append((offset + 1, 45))
    meal_schedule.append((offset + 5, 70))
    meal_schedule.append((offset + 11, 60))


def disable_blocking_figures():
    plt.ioff()

    def _no_show(*args, **kwargs):
        return None

    plt.show = _no_show


def build_illness_scenario():
    illness_duration_days = ILLNESS_END_DAY - ILLNESS_START_DAY
    return IllnessScenario(
        start_time=START_TIME,
        meal_schedule=meal_schedule,
        illness_start_step=288 * ILLNESS_START_DAY,
        illness_duration_steps=288 * illness_duration_days,
        target_reduction_factor=TARGET_REDUCTION_FACTOR,
        target_rat_multiplier=TARGET_RAT_MULTIPLIER,
        max_glucose_offset_mmol_per_l=MAX_GLUCOSE_OFFSET_MMOL_PER_L,
        ramp_fraction=0.2,
    )


def build_loop_controller():
    return LoopController(
        target=110,
        recommendation_type='automaticBolus',
        use_tdd_settings=True,
        insulin_type='novolog',
    )


def build_adaptive_controller():
    return AdaptiveLoopController(
        target=110,
        recommendation_type='automaticBolus',
        use_tdd_settings=True,
        insulin_type='novolog',
        warmup_days=1,
    )


def illness_factors(scenario, current_time):
    """
    Compute illness physiology modifiers 

    Returns:
        reduction_factor: multiplies insulin sensitivity (p2u)
        rat_multiplier: multiplies glucose appearance (f)
    """
    step_idx = scenario._step_index_5min(current_time)
    intensity = scenario._illness_intensity(step_idx)

    reduction_factor = 1.0 - (1.0 - scenario.target_reduction_factor) * intensity
    rat_multiplier = 1.0 + (scenario.target_rat_multiplier - 1.0) * intensity

    return reduction_factor, rat_multiplier


def run_condition(controller_builder, save_path):
    """
    Run one condition (Loop or Adaptive) with a simulation loop that mirrors
    SimObj.simulate()/simulate() semantics:
      1) controller.reset()
      2) env.reset()
      3) repeated policy(...) -> env.step(...)

    Illness-specific behavior is injected only here (test-local), so core
    files remain unchanged.
    """
    os.makedirs(save_path, exist_ok=True)
    all_results = []
    patient_histories = {}
    illness_traces = {}
    controller_traces = {}

    sensor_for_report = CGMSensor.withName("GuardianRT", seed=CGM_SEED)

    for patient_name in PATIENT_NAMES:
        scenario = build_illness_scenario()
        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName("GuardianRT", seed=CGM_SEED)
        pump = InsulinPump.withName("Insulet")

        env = T1DSimEnv(patient, sensor, pump, scenario)
        controller = controller_builder()

        base_p2u = float(env.patient._params.p2u)
        base_f = float(env.patient._params.f)

        illness_rows = []

        # Reset (IMPORTANT — matches SimObj)
        controller.reset()
        obs, reward, done, info = env.reset()

        while env.time < env.scenario.start_time + timedelta(days=SIM_DAYS):

            # --- Illness computation ---
            reduction_factor, rat_multiplier = illness_factors(scenario, env.time)
            if env.time.hour == 8 and env.time.minute == 0:  # once per day
                logger.warning(
                    "Day check | time=%s | p2u=%.4f (base=%.4f, ratio=%.3f) | reduction_factor=%.3f | autosens_ratio=%.3f",
                    env.time, env.patient._params.p2u, base_p2u,
                    env.patient._params.p2u / base_p2u,
                    reduction_factor,
                    controller.get_current_autosens_ratio(patient_name) if hasattr(controller, "get_current_autosens_ratio") else np.nan
                )

            env.patient._params.p2u = base_p2u * reduction_factor
            env.patient._params.f = base_f * rat_multiplier

            # --- Optional controller introspection ---
            autotune_isf = np.nan
            effective_isf = np.nan
            pump_isf = np.nan

            if hasattr(controller, "get_current_isf"):
                try:
                    autotune_isf = controller.get_current_isf(patient_name)
                except Exception:
                    pass

            if hasattr(controller, "get_current_effective_isf"):
                try:
                    effective_isf = controller.get_current_effective_isf(patient_name)
                except Exception:
                    pass

            if hasattr(controller, "get_current_pump_isf"):
                try:
                    pump_isf = controller.get_current_pump_isf(patient_name)
                except Exception:
                    pass

            illness_rows.append({
                "Time": env.time,
                "reduction_factor": reduction_factor,
                "rat_multiplier": rat_multiplier,
                "p2u_effective": env.patient._params.p2u,
                "p2u_relative": env.patient._params.p2u / base_p2u if base_p2u else np.nan,
                "autotune_isf": autotune_isf,
                "effective_isf": effective_isf,
                "pump_isf": pump_isf,
                "p2u_absolute": env.patient._params.p2u,
            })

            # -----------------------------
            # FORCE SAFE CONTROLLER INPUT
            # -----------------------------
            pname = patient_name

            # HARD STRING GUARANTEE (CRITICAL FIX)
            pname = str(pname)

            # prevent NaN / weird pandas types
            if pname is None or pname.lower() == "nan":
                pname = "unknown_patient"

            meal_val = info.get("meal", 0.0)
            sample_time_val = info.get("sample_time", 5)

            try:
                meal_val = float(meal_val)
                if np.isnan(meal_val) or np.isinf(meal_val):
                    meal_val = 0.0
            except Exception:
                meal_val = 0.0

            action = controller.policy(obs, reward, done, **info)

            obs, reward, done, info = env.step(action)

        # Restore baseline
        env.patient._params.p2u = base_p2u
        env.patient._params.f = base_f

        # Save results
        df_patient = env.show_history()
        df_patient.to_csv(os.path.join(save_path, f"{patient_name}.csv"))

        all_results.append(df_patient)
        patient_histories[patient_name] = df_patient

        illness_df = pd.DataFrame(illness_rows)
        if not illness_df.empty:
            illness_df["Time"] = pd.to_datetime(illness_df["Time"])
            illness_df = illness_df.set_index("Time")
            illness_df.to_csv(os.path.join(save_path, f"{patient_name}_illness_trace.csv"))

        illness_traces[patient_name] = illness_df

        meta = {}
        if hasattr(controller, "get_isf_history"):
            try:
                meta["isf_history"] = controller.get_isf_history(patient_name)
            except Exception:
                meta["isf_history"] = []

        if hasattr(controller, "get_autosens_log"):
            try:
                meta["autosens_log"] = controller.get_autosens_log(patient_name)
            except Exception:
                meta["autosens_log"] = []

        controller_traces[patient_name] = meta

    df_all = pd.concat(all_results, keys=PATIENT_NAMES)
    results, _, _, _, _ = report(df_all, sensor_for_report, save_path)

    return results, patient_histories, illness_traces, controller_traces

LOOP_SAVE_PATH = os.path.join(SAVE_PATH, "loop")
ADAPTIVE_SAVE_PATH = os.path.join(SAVE_PATH, "adaptive")
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOOP_SAVE_PATH, exist_ok=True)
os.makedirs(ADAPTIVE_SAVE_PATH, exist_ok=True)

disable_blocking_figures()

print("\nRunning LoopController under illness scenario...")
loop_results, loop_histories, loop_illness_traces, loop_controller_traces = run_condition(build_loop_controller, LOOP_SAVE_PATH)

print("\nRunning AdaptiveLoopController under illness scenario...")
adaptive_results, adaptive_histories, adaptive_illness_traces, adaptive_controller_traces = run_condition(build_adaptive_controller, ADAPTIVE_SAVE_PATH)

print("\n=== Illness scenario comparison (Loop vs Adaptive) ===")
for patient in PATIENT_NAMES:
    loop_row = loop_results.loc[patient]
    adaptive_row = adaptive_results.loc[patient]
    delta_tir = adaptive_row['70<=BG<=180'] - loop_row['70<=BG<=180']
    delta_tar = adaptive_row['BG>180'] - loop_row['BG>180']
    delta_tbr = adaptive_row['BG<70'] - loop_row['BG<70']
    delta_risk = adaptive_row['Risk Index'] - loop_row['Risk Index']

    print(
        f"{patient} | Loop: "
        f"TIR={loop_row['70<=BG<=180']:.2f}%  TAR={loop_row['BG>180']:.2f}%  "
        f"TBR={loop_row['BG<70']:.2f}%  Risk Index={loop_row['Risk Index']:.4f}"
    )
    print(
        f"{patient} | Adaptive: "
        f"TIR={adaptive_row['70<=BG<=180']:.2f}%  TAR={adaptive_row['BG>180']:.2f}%  "
        f"TBR={adaptive_row['BG<70']:.2f}%  Risk Index={adaptive_row['Risk Index']:.4f}"
    )
    print(
        f"{patient} | Delta (Adaptive-Loop): "
        f"TIR={delta_tir:+.2f}%  TAR={delta_tar:+.2f}%  "
        f"TBR={delta_tbr:+.2f}%  Risk Index={delta_risk:+.4f}"
    )


def save_illness_profile_plot(save_path):
    profile_scenario = build_illness_scenario()
    total_steps = int((timedelta(days=SIM_DAYS).total_seconds() // 60) // 5)
    step_idx = list(range(total_steps + 1))

    intensity = [profile_scenario._illness_intensity(s) for s in step_idx]
    reduction_factor = [
        1.0 - (1.0 - profile_scenario.target_reduction_factor) * v for v in intensity
    ]
    glucose_offset_mgdl = [profile_scenario.max_glucose_offset_mgdl * v for v in intensity]

    day_axis = [s / 288.0 for s in step_idx]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(day_axis, intensity, label='Illness intensity', linewidth=2)
    ax1.plot(day_axis, reduction_factor, label='Reduction factor', linewidth=2)
    ax1.set_xlabel('Simulation day')
    ax1.set_ylabel('Unitless scale')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(day_axis, glucose_offset_mgdl, label='Glucose offset (mg/dL)', linewidth=2, linestyle='--')
    ax2.set_ylabel('Glucose offset (mg/dL)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plot_path = os.path.join(save_path, 'illness_profile.png')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def create_combined_metrics(loop_results_df, adaptive_results_df, loop_histories, adaptive_histories):
    """
    Compute metrics from saved BG history dataframes (same data as the CSV files).
    Uses full simulation window for TIR/TAR/TBR, and illness-window TIR separately.
    """

    def compute_metrics_from_df(df):
        if df is None or df.empty or "BG" not in df.columns:
            return {"TIR": np.nan, "TAR": np.nan, "TBR": np.nan}
        bg = df["BG"]
        return {
            "TIR": 100.0 * ((bg >= 70) & (bg <= 180)).mean(),
            "TAR": 100.0 * (bg > 180).mean(),
            "TBR": 100.0 * (bg < 70).mean(),
        }
    """
    def tir_illness_window(df):
        #TIR computed only over the illness+recovery window.
        if df is None or df.empty or "BG" not in df.columns:
            return np.nan
        illness_start_time = START_TIME + timedelta(days=ILLNESS_START_DAY)
        window = df[df.index >= illness_start_time]
        if window.empty:
            return np.nan
        bg = window["BG"]
        return 100.0 * ((bg >= 70) & (bg <= 180)).mean()
    """

    rows = []
    all_patients = sorted(set(loop_histories.keys()) & set(adaptive_histories.keys()))

    for patient_name in all_patients:
        loop_m    = compute_metrics_from_df(loop_histories.get(patient_name))
        adaptive_m = compute_metrics_from_df(adaptive_histories.get(patient_name))

        #loop_tir_illness    = tir_illness_window(loop_histories.get(patient_name))
        #adaptive_tir_illness = tir_illness_window(adaptive_histories.get(patient_name))

        rows.append({
            "Patient": patient_name,
            # Full simulation metrics (matches standalone CSV script)
            "Loop_TIR":     loop_m["TIR"],
            "Adaptive_TIR": adaptive_m["TIR"],
            "Loop_TAR":     loop_m["TAR"],
            "Adaptive_TAR": adaptive_m["TAR"],
            "Loop_TBR":     loop_m["TBR"],
            "Adaptive_TBR": adaptive_m["TBR"],
            # Illness-window TIR separately (more informative for thesis)
            #"Loop_TIR_illness":     loop_tir_illness,
            #"Adaptive_TIR_illness": adaptive_tir_illness,
            # Deltas (full simulation)
            "Delta_TIR": adaptive_m["TIR"]   - loop_m["TIR"],
            "Delta_TAR": adaptive_m["TAR"]   - loop_m["TAR"],
            "Delta_TBR": adaptive_m["TBR"]   - loop_m["TBR"],
            #"Delta_TIR_illness": adaptive_tir_illness - loop_tir_illness,
        })

    return pd.DataFrame(rows).set_index("Patient").sort_index()

def save_tir_comparison_plot(combined_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, tir_col_loop, tir_col_adaptive, title in [
        (axes[0], "Loop_TIR", "Adaptive_TIR",
         "TIR — Full 14-day simulation"),
        #(axes[1], "Loop_TIR_illness", "Adaptive_TIR_illness",
        # f"TIR — Illness+recovery window (day {ILLNESS_START_DAY}–14)"),
    ]:
        for patient_name, row in combined_df.iterrows():
            loop_val     = row[tir_col_loop]
            adaptive_val = row[tir_col_adaptive]
            if np.isnan(loop_val) or np.isnan(adaptive_val):
                continue
            y = [loop_val, adaptive_val]
            line, = ax.plot([0, 1], y, marker="o", linewidth=1.5, alpha=0.85)
            color = line.get_color()
            ax.text(1.03, adaptive_val, str(patient_name),
                    color=color, va="center", fontsize=8)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Loop", "Adaptive"])
        ax.set_ylabel("Time In Range (70–180 mg/dL), %")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlim(-0.1, 1.35)
        ax.set_ylim(0, 105)

    fig.tight_layout()
    plot_path = os.path.join(save_path, "tir_before_after.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def save_bg_p2u_isf_plot(save_path, patient_name, loop_df, adaptive_df, adaptive_illness_df, adaptive_meta):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # --- Panel 1: Loop BG ---
    axes[0].plot(loop_df.index, loop_df["BG"], color="tab:blue", linewidth=1.2)
    axes[0].axhline(70,  color="red",   linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].axhline(180, color="orange",linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].set_title(f"{patient_name} | Loop: BG over time")
    axes[0].set_ylabel("BG (mg/dL)")
    axes[0].grid(True, alpha=0.25)

    # --- Panel 2: Adaptive BG ---
    axes[1].plot(adaptive_df.index, adaptive_df["BG"], color="tab:orange", linewidth=1.2)
    axes[1].axhline(70,  color="red",   linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].axhline(180, color="orange",linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].set_title(f"{patient_name} | Adaptive: BG over time")
    axes[1].set_ylabel("BG (mg/dL)")
    axes[1].grid(True, alpha=0.25)

    # --- Panel 3: ISF comparison + illness signal ---
    ax3 = axes[2]
    ax3.set_title(f"{patient_name} | ISF layers vs illness physiology")
    ax3.set_ylabel("ISF (mg/dL per U)")
    ax3.grid(True, alpha=0.25)

    if adaptive_illness_df is not None and not adaptive_illness_df.empty:

        # Pump ISF — flat baseline reference
        if "pump_isf" in adaptive_illness_df.columns:
            pump_series = adaptive_illness_df["pump_isf"].astype(float)
            if pump_series.notna().any():
                ax3.plot(
                    adaptive_illness_df.index,
                    pump_series,
                    color="gray",
                    linewidth=1.5,
                    linestyle=":",
                    label="Pump ISF (baseline, never changes)",
                )

        # Autotune ISF — slow daily adaptation
        if "autotune_isf" in adaptive_illness_df.columns:
            autotune_series = adaptive_illness_df["autotune_isf"].astype(float)
            if autotune_series.notna().any():
                ax3.plot(
                    adaptive_illness_df.index,
                    autotune_series,
                    color="tab:red",
                    linewidth=1.5,
                    linestyle="-",
                    label="Autotune ISF (daily, permanent)",
                )

        # Effective ISF — fast autosens-scaled value actually used for dosing
        if "effective_isf" in adaptive_illness_df.columns:
            effective_series = adaptive_illness_df["effective_isf"].astype(float)
            if effective_series.notna().any():
                ax3.plot(
                    adaptive_illness_df.index,
                    effective_series,
                    color="tab:purple",
                    linewidth=1.5,
                    linestyle="-",
                    label="Effective ISF = autotune / autosens ratio (used for dosing)",
                )

        # Illness signal on secondary axis
        ax3b = ax3.twinx()
        ax3b.set_ylabel("Relative insulin sensitivity (p2u / baseline)", color="tab:green")
        ax3b.tick_params(axis="y", labelcolor="tab:green")

        if "p2u_relative" in adaptive_illness_df.columns:
            p2u_series = adaptive_illness_df["p2u_relative"].astype(float)
            ax3b.plot(
                adaptive_illness_df.index,
                p2u_series,
                color="tab:green",
                linewidth=2.0,
                linestyle="--",
                alpha=0.7,
                label="p2u / baseline (illness severity, 1=healthy, 0.6=peak illness)",
            )
            ax3b.set_ylim(0, 1.3)

        # Combine legends from both axes
        lines_left, labels_left = ax3.get_legend_handles_labels()
        lines_right, labels_right = ax3b.get_legend_handles_labels()
        ax3.legend(
            lines_left + lines_right,
            labels_left + labels_right,
            loc="lower left",
            fontsize=8,
        )
    else:
        ax3.text(0.5, 0.5, "No illness trace data available",
                 transform=ax3.transAxes, ha="center", va="center")

    axes[2].set_xlabel("Time")
    fig.tight_layout()

    safe_name = str(patient_name).replace("#", "_")
    plot_path = os.path.join(save_path, f"bg_isf_illness_{safe_name}.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path

profile_plot_path = save_illness_profile_plot(SAVE_PATH)
combined_df = create_combined_metrics(loop_results, adaptive_results, loop_histories, adaptive_histories)
combined_csv_path = os.path.join(SAVE_PATH, "combined_metrics.csv")
combined_df.to_csv(combined_csv_path)
tir_plot_path = save_tir_comparison_plot(combined_df, SAVE_PATH)
bg_p2u_isf_plot_paths = []
for patient_name in PATIENT_NAMES:
    loop_df = loop_histories.get(patient_name)
    adaptive_df = adaptive_histories.get(patient_name)
    adaptive_illness_df = adaptive_illness_traces.get(patient_name)
    adaptive_meta = adaptive_controller_traces.get(patient_name, {})
    if loop_df is None or adaptive_df is None:
        continue
    bg_p2u_isf_plot_paths.append(
        save_bg_p2u_isf_plot(
            SAVE_PATH,
            patient_name,
            loop_df,
            adaptive_df,
            adaptive_illness_df,
            adaptive_meta,
        )
    )

summary_path = os.path.join(SAVE_PATH, "summary.txt")
summary_lines = [
    "Illness Scenario Test (Loop vs Adaptive)",
    "=" * 80,
    f"Patients: {', '.join(PATIENT_NAMES)}",
    "Controllers: LoopController, AdaptiveLoopController",
    "",
]

for patient in PATIENT_NAMES:
    loop_row = loop_results.loc[patient]
    adaptive_row = adaptive_results.loc[patient]
    combined_row = combined_df.loc[patient]
    summary_lines.append(
        f"{patient} | Loop: "
        f"TIR={combined_row['Loop_TIR']:.2f}%  TAR={loop_row['BG>180']:.2f}%  "
        f"TBR={loop_row['BG<70']:.2f}%  Risk Index={loop_row['Risk Index']:.4f}"
    )
    summary_lines.append(
        f"{patient} | Adaptive: "
        f"TIR={combined_row['Adaptive_TIR']:.2f}%  TAR={adaptive_row['BG>180']:.2f}%  "
        f"TBR={adaptive_row['BG<70']:.2f}%  Risk Index={adaptive_row['Risk Index']:.4f}"
    )
    summary_lines.append(
        f"{patient} | Delta (Adaptive-Loop): "
        f"TIR={combined_row['Delta_TIR']:+.2f}%  "
        f"TAR={adaptive_row['BG>180'] - loop_row['BG>180']:+.2f}%  "
        f"TBR={adaptive_row['BG<70'] - loop_row['BG<70']:+.2f}%  "
        f"Risk Index={adaptive_row['Risk Index'] - loop_row['Risk Index']:+.4f}"
    )
    summary_lines.append("")

summary_lines.extend([
    "",
    "Combined comparison metrics:",
    combined_df.to_string(float_format=lambda value: f"{value:.4f}"),
    "",
    "Full Loop metrics table:",
    pd.DataFrame(loop_results).to_string(),
    "",
    "Full Adaptive metrics table:",
    pd.DataFrame(adaptive_results).to_string(),
    "",
    f"TIR comparison plot: {tir_plot_path}",
    f"BG/p2u/ISF plots: {', '.join(bg_p2u_isf_plot_paths)}",
    f"Combined metrics CSV: {combined_csv_path}",
    f"Illness profile plot: {profile_plot_path}",
    f"Loop results folder: {LOOP_SAVE_PATH}",
    f"Adaptive results folder: {ADAPTIVE_SAVE_PATH}",
])

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")

print(f"\nSummary saved to: {summary_path}")
