"""Plot adaptive-controller state alongside plant outputs.

Examples:
    python AdaptiveMatlab/plot_adaptations.py

Edit RESULTS_FOLDER below to select a different results folder.

The input folder may contain result CSV files directly or in configuration
subfolders. Controller-state JSON files are optional, so basal-bolus baseline
CSV files can be plotted too.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAMPLE_MIN = 5
RESULTS_FOLDER = (
    Path(__file__).resolve().parents[1]
    / "simglucose_results"
    / "adaptive_matlabv4"
)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def result_identity(csv_path, results_folder):
    patient = csv_path.stem
    relative_parent = csv_path.parent.relative_to(results_folder)
    config = str(relative_parent) if str(relative_parent) != "." else results_folder.name
    return patient, config


def load_state(csv_path):
    state_path = csv_path.with_name(f"{csv_path.stem}_controller_state.json")
    if not state_path.exists():
        return None
    with state_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def indexed_frame(values, columns):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return pd.DataFrame(columns=columns)
    array = np.atleast_2d(array)
    width = min(array.shape[1], len(columns))
    return pd.DataFrame(array[:, :width], columns=columns[:width])


def sample_time(result):
    if "Time" in result:
        timestamps = pd.to_datetime(result["Time"])
        return (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600
    return np.arange(len(result)) * SAMPLE_MIN / 60


def plot_result(csv_path, results_folder, output_folder):
    result = pd.read_csv(csv_path)
    if "BG" not in result:
        return None

    patient, config = result_identity(csv_path, results_folder)
    state = load_state(csv_path)
    hours = sample_time(result)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.suptitle(f"{patient} / {config}")

    adaptation_columns = state.get("adaptationColumns", []) if state else []
    adaptation = indexed_frame(state.get("adaptation", []), adaptation_columns) if state else pd.DataFrame()
    if not adaptation.empty and "step" in adaptation:
        adaptation_hours = adaptation["step"] * SAMPLE_MIN / 60
        left = axes[0, 0]
        for column in ("theta", "Umax"):
            if column in adaptation:
                left.plot(adaptation_hours, adaptation[column], marker="o", label=column)
        left.set_title("Adaptation model")
        left.set_xlabel("Simulation time (hours)")
        left.set_ylabel("Value")
        left.legend()
        left.grid(alpha=0.25)

        pid_columns = [column for column in ("Kp", "Ki", "Kd") if column in adaptation]
        for column in pid_columns:
            axes[0, 1].plot(adaptation_hours, adaptation[column], marker="o", label=column)
        axes[0, 1].set_title("Adapted PID parameters")
        axes[0, 1].set_xlabel("Simulation time (hours)")
        axes[0, 1].set_ylabel("Value")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.25)
    else:
        for axis in axes[0]:
            axis.text(0.5, 0.5, "No adaptation state saved", ha="center", va="center")
            axis.set_axis_off()

    plant_axis = axes[1, 0]
    plant_axis.plot(hours, result["BG"], label="BG", linewidth=0.8)
    if "CGM" in result:
        plant_axis.plot(hours, result["CGM"], label="CGM", linewidth=0.7, alpha=0.7)
    plant_axis.axhspan(70, 180, color="tab:green", alpha=0.08)
    plant_axis.set_title("Plant response")
    plant_axis.set_xlabel("Simulation time (hours)")
    plant_axis.set_ylabel("Glucose (mg/dL)")
    plant_axis.grid(alpha=0.25)
    plant_axis.legend()

    insulin_axis = axes[1, 1]
    if "insulin" in result:
        insulin_axis.plot(hours, result["insulin"], color="tab:red", label="insulin")
    auto_columns = state.get("autoCorrectionColumns", []) if state else []
    auto = indexed_frame(state.get("autoCorrection", []), auto_columns) if state else pd.DataFrame()
    if not auto.empty and "step" in auto:
        auto_hours = auto["step"] * SAMPLE_MIN / 60
        for column in ("bolusU", "capped", "lowThreshold"):
            if column in auto:
                insulin_axis.plot(auto_hours, auto[column], marker=".", linewidth=0.8, label=column)
    insulin_axis.set_title("Insulin and auto-correction")
    insulin_axis.set_xlabel("Simulation time (hours)")
    insulin_axis.set_ylabel("Insulin / correction value")
    insulin_axis.grid(alpha=0.25)
    insulin_axis.legend()

    figure.tight_layout()
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{safe_name(config)}_{safe_name(patient)}_adaptations.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main():
    results_folder = RESULTS_FOLDER.resolve()
    output_folder = results_folder / "adaptation_plots"
    csv_paths = sorted(
        path for path in results_folder.rglob("*.csv")
        if path.name not in {"summary.csv"} and "BG" in pd.read_csv(path, nrows=0).columns
    )
    if not csv_paths:
        raise FileNotFoundError(
            f"No simulation CSV files with a BG column found in {results_folder}"
        )

    outputs = [plot_result(path, results_folder, output_folder) for path in csv_paths]
    for output in outputs:
        if output is not None:
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
