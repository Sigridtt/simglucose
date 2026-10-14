from datetime import datetime, timedelta
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

# The bridge is in the separate AdaptiveControllerMatLab repository.
BRIDGE_REPO = Path(
    r"C:\Users\sigridtt\OneDrive - NTNU\Documents\Master\AdaptiveControllerMatLab"
)
if str(BRIDGE_REPO) not in sys.path:
    sys.path.insert(0, str(BRIDGE_REPO))

from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj
from simglucose_bridge import AdaptiveControllerV3, start_engine

SIMGLUCOSE_ROOT = Path(__file__).resolve().parents[1]
PATIENT_FILE = SIMGLUCOSE_ROOT / "simglucose" / "params" / "vpatient_params.csv"
OUTPUT_DIR = SIMGLUCOSE_ROOT / "simglucose_results" / "adaptive_matlab"

PATIENTS = ["child#001", "adolescent#001", "adult#001"]
SCENARIOS = {
    "standard": [(7, 45), (12, 70), (18, 80)],
    "small_meals": [(7, 30), (12, 40), (18, 50)],
    "no_meals": [],
}


def run_all(days=3, seed=1):
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
                controller = AdaptiveControllerV3(engine)
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
