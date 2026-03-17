from datetime import datetime, timedelta
from simglucose.simulation.user_interface import simulate
from simglucose.controller.loop_ctrller import LoopController
from simglucose.simulation.scenario import CustomScenario
import os
import pandas as pd
import sys
sys.dont_write_bytecode = True  # avoid .pyc files to prevent confusion with the "real" Loop code in the LoopToPython package
START_TIME = datetime(2018, 1, 1, 8, 0, 0)

# 7 days of meals: breakfast, lunch, dinner each day
meals = []
for day in range(7):
    offset = day * 24
    meals.append((offset + 1,  45))  # breakfast
    meals.append((offset + 5,  70))  # lunch
    meals.append((offset + 11, 60))  # dinner

scenario = CustomScenario(start_time=START_TIME, scenario=meals)

controller = LoopController(
    target=110,
    recommendation_type='automaticBolus',
    use_tdd_settings=True,
    insulin_type='novolog',
)

SAVE_PATH = os.path.join(os.path.dirname(__file__), "examples", "results", "test_loop_7day")
os.makedirs(SAVE_PATH, exist_ok=True)

results = simulate(
    sim_time=timedelta(days=7),
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

# --- Build summary ---
summary_lines = [
    "=== Results: adult#001, 7 days (LoopController) ===",
    f"  Time in range (70-180) : {results['70<=BG<=180'].values[0]:.1f} %",
    f"  Time above 180         : {results['BG>180'].values[0]:.1f} %",
    f"  Time below 70          : {results['BG<70'].values[0]:.1f} %",
    f"  Risk Index             : {results['Risk Index'].values[0]:.4f}",
]

# --- Print to terminal ---
print()
for line in summary_lines:
    print(line)

# --- Save summary as .txt ---
summary_path = os.path.join(SAVE_PATH, "summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"\n  Summary saved to: {summary_path}")

# --- Save full results as .csv ---
results_csv_path = os.path.join(SAVE_PATH, "results.csv")
results.to_csv(results_csv_path)
print(f"  Full results saved to: {results_csv_path}")