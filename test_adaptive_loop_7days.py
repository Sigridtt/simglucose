from datetime import datetime, timedelta
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.analysis.report import report
from simglucose.controller.adaptive_loop_ctrller import AdaptiveLoopController
import pandas as pd
import os
import sys
sys.dont_write_bytecode = True  # avoid .pyc files to prevent confusion with the "real" Loop code in the LoopToPython package
START_TIME = datetime(2018, 1, 1, 8, 0, 0)
PATIENT_NAME = "adult#001"

# 7 days of meals
meals = []
for day in range(7):
    offset = day * 24
    meals.append((offset + 1,  45))
    meals.append((offset + 5,  70))
    meals.append((offset + 11, 60))

scenario = CustomScenario(start_time=START_TIME, scenario=meals)

controller = AdaptiveLoopController(
    target=110,
    recommendation_type='automaticBolus',
    use_tdd_settings=True,
    insulin_type='novolog',
    warmup_days=1,
    adaptation_interval_hours=24,
    n_autotune_iterations=1,
)

SAVE_PATH = os.path.join(os.path.dirname(__file__), "examples", "results", "test_adaptive_7day")
os.makedirs(SAVE_PATH, exist_ok=True)

# Build environment manually — no deepcopy, so controller state is preserved
patient    = T1DPatient.withName(PATIENT_NAME)
sensor     = CGMSensor.withName('GuardianRT', seed=1)
pump       = InsulinPump.withName('Insulet')
env        = T1DSimEnv(patient, sensor, pump, scenario)
sim_object = SimObj(env, controller, timedelta(days=7), animate=False, path=SAVE_PATH)

import time
t0 = time.time()
print("Simulation starts ...")
result_df = sim(sim_object)
print(f"Simulation Completed!")
print(f"Simulation took {time.time() - t0:.2f} sec.")

# Report
all_df = pd.concat([result_df], keys=[PATIENT_NAME])
results, ri_per_hour, zone_stats, figs, axes = report(all_df, sensor, SAVE_PATH)

# --- Build summary ---
summary_lines = [
    "=== Results: adult#001, 7 days (Adaptive ISF) ===",
    f"  Time in range (70-180) : {results['70<=BG<=180'].values[0]:.1f} %",
    f"  Time above 180         : {results['BG>180'].values[0]:.1f} %",
    f"  Time below 70          : {results['BG<70'].values[0]:.1f} %",
    f"  Risk Index             : {results['Risk Index'].values[0]:.4f}",
    "",
    "=== ISF adaptation history ===",
]

isf_history = controller.get_isf_history(PATIENT_NAME)
if isf_history:
    for entry in isf_history:
        line = f"  {entry['datetime']}  ISF {entry['old_isf']:.3f} -> {entry['new_isf']:.3f} mg/dL/U"
        summary_lines.append(line)
else:
    summary_lines.append("  No ISF adaptations occurred.")

print()
for line in summary_lines:
    print(line)

# Save summary
summary_path = os.path.join(SAVE_PATH, "summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"\n  Summary saved to: {summary_path}")

# Save ISF history
if isf_history:
    isf_df = pd.DataFrame(isf_history)
    isf_csv_path = os.path.join(SAVE_PATH, "isf_history.csv")
    isf_df.to_csv(isf_csv_path, index=False)
    print(f"  ISF history saved to: {isf_csv_path}")

# Save results
results_csv_path = os.path.join(SAVE_PATH, "results.csv")
results.to_csv(results_csv_path)
print(f"  Full results saved to: {results_csv_path}")