from datetime import datetime, timedelta
from simglucose.simulation.user_interface import simulate
from simglucose.controller.loop_ctrller import LoopController
from simglucose.simulation.scenario import CustomScenario
import os
import sys
sys.dont_write_bytecode = True  # avoid .pyc files to prevent confusion with the "real" Loop code in the LoopToPython package
START_TIME = datetime(2018, 1, 1, 8, 0, 0)

scenario = CustomScenario(
    start_time=START_TIME,
    scenario=[(1, 45)], # meal at 1 hour in = 9am
)

controller = LoopController(
    target=110,
    recommendation_type='automaticBolus',
    use_tdd_settings=True,
    insulin_type='novolog',
)
# Save results to a local folder
SAVE_PATH = os.path.join(os.path.dirname(__file__), "examples", "results", "test_run")
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



print("\n=== Results: adult#001, 24h ===")
print(f"  Time in range (70-180) : {results['70<=BG<=180'].values[0]:.1f} %")
print(f"  Time above 180         : {results['BG>180'].values[0]:.1f} %")
print(f"  Time below 70          : {results['BG<70'].values[0]:.1f} %")
print(f"  LBGI                   : {results['LBGI'].values[0]:.4f}")
print(f"  HBGI                   : {results['HBGI'].values[0]:.4f}")
print(f"  Risk Index             : {results['Risk Index'].values[0]:.4f}")