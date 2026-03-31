from datetime import datetime, timedelta
import os
import sys
import logging
import pkg_resources
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simglucose.simulation.user_interface import simulate
from simglucose.controller.loop_ctrller import LoopController
from simglucose.controller.adaptive_loop_ctrller import AdaptiveLoopController
from simglucose.simulation.scenario import CustomScenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

sys.dont_write_bytecode = True

START_TIME = datetime(2018, 1, 1, 8, 0, 0)
SIM_DURATION = timedelta(days=7)
PATIENT_PARA_FILE = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")

BASE_SAVE_PATH = os.path.join(
    os.path.dirname(__file__),
    "examples",
    "results",
    "all_patients_7days_compared",
)
LOOP_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "loop")
ADAPTIVE_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "adaptive")
os.makedirs(LOOP_SAVE_PATH, exist_ok=True)
os.makedirs(ADAPTIVE_SAVE_PATH, exist_ok=True)


def disable_blocking_figures():
    plt.ioff()

    def _no_show(*args, **kwargs):
        return None

    plt.show = _no_show


def build_7day_scenario():
    meals = []
    for day in range(7):
        offset = day * 24
        meals.append((offset + 1, 45))
        meals.append((offset + 5, 70))
        meals.append((offset + 11, 60))
    return CustomScenario(start_time=START_TIME, scenario=meals)


def get_all_patients():
    patient_df = pd.read_csv(PATIENT_PARA_FILE)
    return list(patient_df["Name"].values)


def run_loop(patient_names, scenario):
    controller = LoopController(
        target=110,
        recommendation_type="automaticBolus",
        use_tdd_settings=True,
        insulin_type="novolog",
    )

    return simulate(
        sim_time=SIM_DURATION,
        scenario=scenario,
        controller=controller,
        patient_names=patient_names,
        cgm_name="GuardianRT",
        cgm_seed=1,
        insulin_pump_name="Insulet",
        start_time=START_TIME,
        save_path=LOOP_SAVE_PATH,
        animate=False,
        parallel=False,
    )


def run_adaptive(patient_names, scenario):
    controller = AdaptiveLoopController(
        target=110,
        recommendation_type="automaticBolus",
        use_tdd_settings=True,
        insulin_type="novolog",
        warmup_days=1,
        adaptation_interval_hours=24,
        n_autotune_iterations=1,
        debug_timing=False,
    )

    return simulate(
        sim_time=SIM_DURATION,
        scenario=scenario,
        controller=controller,
        patient_names=patient_names,
        cgm_name="GuardianRT",
        cgm_seed=1,
        insulin_pump_name="Insulet",
        start_time=START_TIME,
        save_path=ADAPTIVE_SAVE_PATH,
        animate=False,
        parallel=False,
    )


def create_combined_metrics(loop_results, adaptive_results):
    loop_df = pd.DataFrame(loop_results).copy()
    adaptive_df = pd.DataFrame(adaptive_results).copy()

    loop_cols = {
        "70<=BG<=180": "Loop_TIR",
        "BG>180": "Loop_TAR",
        "BG<70": "Loop_TBR",
        "Risk Index": "Loop_RiskIndex",
    }
    adaptive_cols = {
        "70<=BG<=180": "Adaptive_TIR",
        "BG>180": "Adaptive_TAR",
        "BG<70": "Adaptive_TBR",
        "Risk Index": "Adaptive_RiskIndex",
    }

    merged = pd.concat(
        [
            loop_df[list(loop_cols.keys())].rename(columns=loop_cols),
            adaptive_df[list(adaptive_cols.keys())].rename(columns=adaptive_cols),
        ],
        axis=1,
    )

    merged["Delta_TIR"] = merged["Adaptive_TIR"] - merged["Loop_TIR"]
    merged["Delta_TAR"] = merged["Adaptive_TAR"] - merged["Loop_TAR"]
    merged["Delta_TBR"] = merged["Adaptive_TBR"] - merged["Loop_TBR"]
    merged["Delta_RiskIndex"] = merged["Adaptive_RiskIndex"] - merged["Loop_RiskIndex"]
    return merged.sort_index()


def save_tir_comparison_plot(combined_df):
    fig, ax = plt.subplots(figsize=(11, 8))
    x = [0, 1]

    for patient_name, row in combined_df.iterrows():
        y = [row["Loop_TIR"], row["Adaptive_TIR"]]
        ax.plot(x, y, marker="o", linewidth=1.2, alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(["Loop", "Adaptive"])
    ax.set_ylabel("Time In Range (70-180 mg/dL), %")
    ax.set_title("TIR Before vs After Adaptation (All Patients, 7 days)")
    ax.grid(True, axis="y", alpha=0.3)

    plot_path = os.path.join(BASE_SAVE_PATH, "tir_before_after_all_patients.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def write_summary(combined_df, patient_names, plot_path):
    summary_path = os.path.join(BASE_SAVE_PATH, "summary_all_patients.txt")

    lines = [
        "7-day Comparison: Loop vs Adaptive (All Patients)",
        "=" * 90,
        f"Total patients: {len(patient_names)}",
        f"Patients: {', '.join(patient_names)}",
        "",
        "Aggregate means:",
        (
            f"Loop      TIR={combined_df['Loop_TIR'].mean():.2f}%  "
            f"TAR={combined_df['Loop_TAR'].mean():.2f}%  "
            f"TBR={combined_df['Loop_TBR'].mean():.2f}%  "
            f"Risk Index={combined_df['Loop_RiskIndex'].mean():.4f}"
        ),
        (
            f"Adaptive  TIR={combined_df['Adaptive_TIR'].mean():.2f}%  "
            f"TAR={combined_df['Adaptive_TAR'].mean():.2f}%  "
            f"TBR={combined_df['Adaptive_TBR'].mean():.2f}%  "
            f"Risk Index={combined_df['Adaptive_RiskIndex'].mean():.4f}"
        ),
        (
            f"Delta (Adaptive-Loop): TIR={combined_df['Delta_TIR'].mean():+.2f}%  "
            f"TAR={combined_df['Delta_TAR'].mean():+.2f}%  "
            f"TBR={combined_df['Delta_TBR'].mean():+.2f}%  "
            f"Risk Index={combined_df['Delta_RiskIndex'].mean():+.4f}"
        ),
        "",
        "Per-patient metrics:",
        combined_df.to_string(float_format=lambda value: f"{value:.4f}"),
        "",
        "Saved outputs:",
        f"- Loop folder: {LOOP_SAVE_PATH}",
        f"- Adaptive folder: {ADAPTIVE_SAVE_PATH}",
        f"- Combined CSV: {os.path.join(BASE_SAVE_PATH, 'combined_metrics.csv')}",
        f"- TIR plot: {plot_path}",
    ]

    with open(summary_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines) + "\n")

    return summary_path


def main():
    disable_blocking_figures()

    patient_names = get_all_patients()
    scenario = build_7day_scenario()

    print("Simulation starts for all patients (7 days) ...")
    print(f"Patients ({len(patient_names)}): {', '.join(patient_names)}")

    print("\nRunning LoopController ...")
    loop_results = run_loop(patient_names, scenario)

    print("\nRunning AdaptiveLoopController ...")
    adaptive_results = run_adaptive(patient_names, scenario)

    combined_df = create_combined_metrics(loop_results, adaptive_results)

    combined_csv_path = os.path.join(BASE_SAVE_PATH, "combined_metrics.csv")
    combined_df.to_csv(combined_csv_path)

    plot_path = save_tir_comparison_plot(combined_df)
    summary_path = write_summary(combined_df, patient_names, plot_path)

    print("\nFinished all simulations.")
    print(f"Combined metrics saved to: {combined_csv_path}")
    print(f"TIR comparison plot saved to: {plot_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
