"""Run one patient for one day and plot BG, IOB, and the MATLAB CoB profile.

IOB and CoB are read directly from the MATLAB controller's compact state
export. They are not recomputed in Python.

Run from the simglucose_git directory::

    python AdaptiveMatlab/test_one_meal_plot.py
    python AdaptiveMatlab/test_one_meal_plot.py --patient adult#001
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import first_test

from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "simglucose_results" / "one_meal_plot_IFB"
PATIENT = "adult#001"
MEAL_HOUR = 12.0
MEAL_GRAMS = 40.0
DAYS = 1
SAMPLE_MINUTES = 5
CONTROLLER_NAME = "adaptive780G"
#CONTROLLER_NAME = "adaptiveV3"
#CONTROLLER_NAME = "adaptiveV4"


def controller_board_state(state):
    steps = np.asarray(state.get("step", []), dtype=float).reshape(-1)
    cob = np.asarray(state.get("COB", []), dtype=float).reshape(-1)
    iob = np.asarray(state.get("IOBNet", []), dtype=float).reshape(-1)
    if not (len(steps) == len(cob) == len(iob)):
        raise ValueError("MATLAB controller state has inconsistent step/COB/IOB lengths")
    return steps * SAMPLE_MINUTES / 60.0, cob, iob


def meal_bolus_hours(state):
    steps = np.asarray(state.get("step", []), dtype=float).reshape(-1)
    bolus_type = np.asarray(state.get("bolusType", []), dtype=float).reshape(-1)
    bolus = np.asarray(state.get("bolusU", []), dtype=float).reshape(-1)
    meal_bolus = (bolus_type == 1) & (bolus > 0)
    return steps[meal_bolus] * SAMPLE_MINUTES / 60.0


def run(patient, seed):
    from simglucose_bridge import AdaptiveController, start_engine

    engine = start_engine("matlab", str(first_test.BRIDGE_REPO))
    try:
        env = T1DSimEnv(
            T1DPatient.withName(patient),
            CGMSensor.withName("GuardianRT", seed=seed),
            InsulinPump.withName("Insulet"),
            CustomScenario(
                start_time=datetime(2020, 1, 1),
                scenario=[(MEAL_HOUR, MEAL_GRAMS)],
            ),
        )
        controller = AdaptiveController(
            engine,
            adaptation=False,
            announce="observed",
            controller=CONTROLLER_NAME,
        )
        simulation = SimObj(env, controller, timedelta(days=DAYS), animate=False)
        simulation.simulate()
        result = simulation.results()
        state = first_test.state_to_jsonable(controller.controller_state())
        return result, state
    finally:
        engine.close()


def plot(result, state, patient):
    if isinstance(result.index, pd.DatetimeIndex):
        hours = (result.index - result.index[0]).total_seconds() / 3600.0
    elif "Time" in result:
        timestamps = pd.to_datetime(result["Time"])
        hours = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    else:
        hours = np.arange(len(result)) * SAMPLE_MINUTES / 60.0

    controller_hours, cob, iob = controller_board_state(state)
    meal_bolus_hours_ = meal_bolus_hours(state)

    figure, (bg_axis, board_axis) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    bg_axis.plot(hours, result["BG"], color="tab:blue", linewidth=1.4, label="BG")
    bg_axis.axvline(MEAL_HOUR, color="tab:orange", linestyle="--", label="40 g meal")
    for index, bolus_hour in enumerate(meal_bolus_hours_):
        bg_axis.axvline(
            bolus_hour,
            color="tab:red",
            linestyle=":",
            label="meal bolus" if index == 0 else None,
        )
    bg_axis.axhspan(70, 180, color="tab:green", alpha=0.08)
    bg_axis.set_ylabel("Blood glucose (mg/dL)")
    bg_axis.set_title(f"{patient}: one-day 40 g meal at 12:00")
    bg_axis.grid(alpha=0.25)
    bg_axis.legend(loc="upper left")

    board_axis.plot(controller_hours, iob, color="tab:red", label="IOB net (MATLAB)")
    board_axis.plot(controller_hours, cob, color="tab:green", label="CoB (MATLAB)")
    board_axis.axvline(MEAL_HOUR, color="tab:orange", linestyle="--", label="Meal time")
    for index, bolus_hour in enumerate(meal_bolus_hours_):
        board_axis.axvline(
            bolus_hour,
            color="tab:red",
            linestyle=":",
            label="Meal bolus" if index == 0 else None,
        )
    board_axis.set_xlabel("Simulation time (hours)")
    board_axis.set_ylabel("Units / grams")
    board_axis.grid(alpha=0.25)
    board_axis.legend(loc="upper right")
    board_axis.set_xlim(0, 24)
    figure.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{patient}_one_meal.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default=PATIENT)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    result, state = run(args.patient, args.seed)
    output_path = plot(result, state, args.patient)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()