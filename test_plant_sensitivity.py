"""Compare simglucose plant sensitivity with the adaptive controller model.

The experiment runs three independent, meal-free simulations at basal insulin,
basal minus a step, and basal plus a step. The last ``steady_hours`` of each
run are averaged and the plant gain is estimated with a central difference.

Example::

    python test_plant_sensitivity.py --patient adult#001 \
        --state-json simglucose_results/adaptive_matlab_v3_ablation_TDD_off/full/adult#001_controller_state.json \
        --model-gain -60

``--model-gain`` is the controller's nominal ``Gain0`` in mmol/L per U/min.
The controller applies sensitivity as ``Gain(theta) = Gain0 * theta``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action, Controller
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj


class FixedBasalController(Controller):
    """Deliver one fixed basal rate and no bolus."""

    def __init__(self, basal_u_per_min: float):
        super().__init__(init_state=None)
        self.basal_u_per_min = basal_u_per_min

    def reset(self):
        return None

    def policy(self, observation, reward, done, **info):
        return Action(basal=self.basal_u_per_min, bolus=0.0)


def patient_basal(patient: str) -> float:
    patient_obj = T1DPatient.withName(patient)
    return float(patient_obj._params.u2ss * patient_obj._params.BW / 6000.0)


def run_step(patient: str, insulin_u_per_min: float, days: int, seed: int) -> np.ndarray:
    env = T1DSimEnv(
        T1DPatient.withName(patient),
        CGMSensor.withName("GuardianRT", seed=seed),
        InsulinPump.withName("Insulet"),
        CustomScenario(start_time=datetime(2020, 1, 1), scenario=[]),
    )
    simulation = SimObj(
        env,
        FixedBasalController(insulin_u_per_min),
        timedelta(days=days),
        animate=False,
    )
    simulation.simulate()
    return simulation.results()["BG"].to_numpy(dtype=float)


def steady_state_gain(
    low_bg: np.ndarray,
    high_bg: np.ndarray,
    low_insulin: float,
    high_insulin: float,
    samples: int,
) -> float:
    """Return plant gain in mmol/L per U/min; more insulin should lower BG."""
    low = float(np.mean(low_bg[-samples:]))
    high = float(np.mean(high_bg[-samples:]))
    return (high - low) / (high_insulin - low_insulin) / 18.0


def load_theta_hat(path: Path) -> float:
    state = json.loads(path.read_text(encoding="utf-8"))
    values = np.asarray(state.get("thetaHat", []), dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"No finite thetaHat values in {path}")
    return float(values[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="adult#001")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--steady-hours", type=float, default=12.0)
    parser.add_argument("--step-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-gain", type=float, required=True)
    parser.add_argument("--theta-hat", type=float)
    parser.add_argument("--state-json", type=Path)
    args = parser.parse_args(argv)

    if args.theta_hat is None and args.state_json is None:
        parser.error("provide --theta-hat or --state-json")
    theta_hat = args.theta_hat if args.theta_hat is not None else load_theta_hat(args.state_json)
    if args.days * 24 <= args.steady_hours:
        parser.error("--days must be longer than --steady-hours")
    if not 0 < args.step_fraction < 1:
        parser.error("--step-fraction must be between 0 and 1")

    basal = patient_basal(args.patient)
    low_insulin = basal * (1.0 - args.step_fraction)
    high_insulin = basal * (1.0 + args.step_fraction)
    low_bg = run_step(args.patient, low_insulin, args.days, args.seed)
    high_bg = run_step(args.patient, high_insulin, args.days, args.seed)
    samples = round(args.steady_hours * 60 / 5)
    plant_gain = steady_state_gain(
        low_bg, high_bg, low_insulin, high_insulin, samples
    )
    gain_theta_1 = args.model_gain
    gain_theta_hat = args.model_gain * theta_hat

    print(f"patient:             {args.patient}")
    print(f"basal:               {basal:.6g} U/min")
    print(f"plant gain:          {plant_gain:.6g} mmol/L per U/min")
    print(f"model gain theta=1:  {gain_theta_1:.6g} mmol/L per U/min")
    print(f"thetaHat:            {theta_hat:.6g}")
    print(f"model gain thetaHat: {gain_theta_hat:.6g} mmol/L per U/min")
    print(f"plant/model theta=1: {plant_gain / gain_theta_1:.6g}")
    print(f"plant/model thetaHat:{plant_gain / gain_theta_hat:.6g}")


if __name__ == "__main__":
    main()