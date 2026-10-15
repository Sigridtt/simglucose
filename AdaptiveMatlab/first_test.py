from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONTROLLER_REPO_URL = "https://github.com/Sigridtt/AdaptiveControllerMatLab.git"
CONTROLLER_REPO_BRANCH = "test"
CONTROLLER_CACHE = Path(tempfile.gettempdir()) / "AdaptiveControllerMatLab"


def fetch_controller_repo():
    if not (CONTROLLER_CACHE / ".git").exists():
        subprocess.run(
            [
                "git", "clone", "--branch", CONTROLLER_REPO_BRANCH,
                "--single-branch", CONTROLLER_REPO_URL, str(CONTROLLER_CACHE),
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                "git", "-C", str(CONTROLLER_CACHE), "fetch", "origin",
                CONTROLLER_REPO_BRANCH,
            ],
            check=True,
        )
        subprocess.run(
            [
                "git", "-C", str(CONTROLLER_CACHE), "checkout", "-B",
                CONTROLLER_REPO_BRANCH, f"origin/{CONTROLLER_REPO_BRANCH}",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git", "-C", str(CONTROLLER_CACHE), "reset", "--hard",
                f"origin/{CONTROLLER_REPO_BRANCH}",
            ],
            check=True,
        )
    return CONTROLLER_CACHE


# The bridge is fetched from the separate AdaptiveControllerMatLab repository.
BRIDGE_REPO = fetch_controller_repo()
if str(BRIDGE_REPO) not in sys.path:
    sys.path.insert(0, str(BRIDGE_REPO))

from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj
from simglucose_bridge import AdaptiveController, start_engine

SIMGLUCOSE_ROOT = Path(__file__).resolve().parents[1]
PATIENT_FILE = SIMGLUCOSE_ROOT / "simglucose" / "params" / "vpatient_params.csv"
OUTPUT_DIR = SIMGLUCOSE_ROOT / "simglucose_results" / "adaptive_matlabv4"

PATIENTS = ["child#001", "adolescent#001", "adult#001"]
DAYS = 14
SEED = 1


def daily_meals(meals, days):
    return [
        (24 * day + hour, grams)
        for day in range(days)
        for hour, grams in meals
    ]


SCENARIOS = {
    "standard": daily_meals([(7, 45), (12, 70), (18, 80)], DAYS),
}


def state_to_jsonable(state):
    field_names = getattr(state, "_fieldnames", None)
    if field_names is None:
        field_names = list(state.keys())
    return {
        field: np.asarray(state[field], dtype=float).reshape(-1).tolist()
        if field in ("step", "bolusType", "aggressiveness", "theta")
        else np.asarray(state[field], dtype=float).tolist()
        for field in field_names
    }


def run_all(days=DAYS, seed=SEED):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    engine = start_engine("matlab", str(BRIDGE_REPO))

    try:
        for patient_name in PATIENTS:
            for scenario_name, meals in SCENARIOS.items():
                print(f"Running {patient_name} / {scenario_name}")
                start_time = datetime(2020, 1, 1, 0, 0, 0)
                scenario = CustomScenario(start_time=start_time, scenario=meals)
                env = T1DSimEnv(
                    T1DPatient.withName(patient_name),
                    CGMSensor.withName("GuardianRT", seed=seed),
                    InsulinPump.withName("Insulet"),
                    scenario,
                )
                controller = AdaptiveController(engine, controller="adaptiveV4")
                simulation = SimObj(
                    env,
                    controller,
                    timedelta(days=days),
                    animate=False,
                )
                simulation.simulate()
                result = simulation.results()
                result.to_csv(
                    OUTPUT_DIR / f"{patient_name}_{scenario_name}.csv",
                    index=False,
                )
                state = controller.controller_state()
                state_json = state_to_jsonable(state)
                state_json["adaptationColumns"] = [
                    "step", "nDays", "theta", "Umax", "Kp", "Ki", "Kd"
                ]
                state_json["autoCorrectionColumns"] = [
                    "step", "cgm", "iob", "raw", "capped", "bolusU",
                    "minBG_noBolus", "minBG_withBolus", "lowThreshold"
                ]
                state_path = OUTPUT_DIR / (
                    f"{patient_name}_{scenario_name}_controller_state.json"
                )
                state_path.write_text(
                    json.dumps(state_json, indent=2),
                    encoding="utf-8",
                )
                results[(patient_name, scenario_name)] = result
    finally:
        engine.close()

    colors = {
        "child#001": "tab:blue",
        "adolescent#001": "tab:orange",
        "adult#001": "tab:green",
    }
    line_styles = {
        "standard": "-",
        "small_meals": "--",
        "no_meals": ":",
    }

    figure, axis = plt.subplots(figsize=(14, 7))
    for (patient_name, scenario_name), result in results.items():
        elapsed_hours = (
            result.index - result.index[0]
        ).total_seconds() / 3600
        axis.plot(
            elapsed_hours,
            result["BG"],
            color=colors[patient_name],
            linestyle=line_styles[scenario_name],
            linewidth=1.5,
            label=f"{patient_name} / {scenario_name}",
        )

    axis.axhline(70, color="black", linewidth=0.8, alpha=0.5)
    axis.axhline(180, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title("Simglucose BGL comparison")
    axis.set_xlabel("Simulation time (hours)")
    axis.set_ylabel("Blood glucose (mg/dL)")
    axis.set_xlim(left=0)
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    plot_path = OUTPUT_DIR / "bgl_comparison.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    print(f"Saved BGL comparison plot to {plot_path}")


if __name__ == "__main__":
    run_all()
