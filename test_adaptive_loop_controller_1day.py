from datetime import datetime, timedelta
from simglucose.simulation.user_interface import simulate
from simglucose.controller.adaptive_loop_ctrller import AdaptiveLoopController
from simglucose.simulation.scenario import CustomScenario
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s - %(message)s'
)

sys.dont_write_bytecode = True  # avoid .pyc files to prevent confusion with the "real" Loop code in the LoopToPython package
START_TIME = datetime(2018, 1, 1, 8, 0, 0)

scenario = CustomScenario(
    start_time=START_TIME,
    scenario=[(1, 45), (5, 70), (11, 60)],  # breakfast, lunch, dinner
)

controller = AdaptiveLoopController(
    target=100,
    recommendation_type='automaticBolus',
    use_tdd_settings=True,
    insulin_type='novolog',
    adaptation_interval_hours=24,
    n_autotune_iterations=1,
    debug_timing=True,
    debug_every_steps=1,
    slow_step_seconds=1.0,
)

SAVE_PATH = os.path.join(os.path.dirname(__file__), "examples", "results", "test_adaptive", "two-layer_1st")
os.makedirs(SAVE_PATH, exist_ok=True)

results = simulate(
    sim_time=timedelta(hours=24),
    scenario=scenario,
    controller=controller,
    patient_names=["adult#001"],
    cgm_name="GuardianRT",
    cgm_seed=1,
    insulin_pump_name="Insulet",
    start_time=START_TIME,
    save_path=SAVE_PATH,
    animate=False,
    parallel=False,
)

print("\n=== Results: adult#001, 24h (Adaptive) ===")
print(f"  Time in range (70-180) : {results['70<=BG<=180'].values[0]:.1f} %")
print(f"  Time above 180         : {results['BG>180'].values[0]:.1f} %")
print(f"  Time below 70          : {results['BG<70'].values[0]:.1f} %")
print(f"  Risk Index             : {results['Risk Index'].values[0]:.4f}")

print("\n=== ISF adaptation history for adult#001 ===")
print(controller.get_isf_history("adult#001"))