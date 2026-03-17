"""
Prints avgDelta, BGI, and deviation for the last autotune window
to visualise the systematic bias.
"""
from datetime import datetime, timedelta
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.adaptive_loop_ctrller import AdaptiveLoopController
from loop_to_python_adaptive.loop_oref_mapping import (
    add_avg_delta_to_history_df,
    add_bgi_to_history_df,
    add_deviation_to_history_df,
)
from loop_to_python_api.api import get_prediction_values_and_dates
import pandas as pd
import os
import sys
sys.dont_write_bytecode = True  # avoid .pyc files to prevent confusion with the "real" Loop code in the LoopToPython package
START_TIME = datetime(2018, 1, 1, 8, 0, 0)
PATIENT_NAME = "adult#001"

SAVE_PATH = os.path.join(os.path.dirname(__file__), "examples", "results", "debug_bias")
os.makedirs(SAVE_PATH, exist_ok=True)

# 2-day run: warmup day 1, autotune fires at start of day 2
# We need a 3-day run so there is still json_history AFTER the day-2 autotune fires
meals = []
for day in range(3):
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
)

patient    = T1DPatient.withName(PATIENT_NAME)
sensor     = CGMSensor.withName('GuardianRT', seed=1)
pump       = InsulinPump.withName('Insulet')
env        = T1DSimEnv(patient, sensor, pump, scenario)
sim_object = SimObj(env, controller, timedelta(days=3), animate=False, path=SAVE_PATH)
sim(sim_object)

# --- Get state after simulation ---
state     = controller._adaptive_state[PATIENT_NAME]
isf_pump  = state['isf_pump']

quest       = controller.quest[controller.quest.Name.str.match(PATIENT_NAME)]
TDD         = quest.TDI.values[0]
basal_pr_hr, isf_pump, cr = controller.get_therapy_settings_from_tdd(TDD)

# Use the merged json from the last adaptation window still in memory
# (day 3 window — not yet adapted)
json_history = state['json_history']
merged_json  = controller._merge_json_inputs(json_history)

print(f"\n  json snapshots in current window : {len(json_history)}")
print(f"  total doses in merged json       : {len(merged_json.get('doses', []))}")

# Use last 24h of observations as the analysis window
df_obs   = controller.observations[PATIENT_NAME].dropna(subset=['CGM'])
df_day   = df_obs.tail(288).copy()

# --- Method A: activity-based BGI (current approach, with merged doses) ---
df_a = add_bgi_to_history_df(df_day, isf_pump, loop_algorithm_input=merged_json, json_history=json_history)
df_a = add_avg_delta_to_history_df(df_a)
df_a = add_deviation_to_history_df(df_a)

# --- Method B: prediction-based BGI (Loop's own output) ---
last_snapshot = state['json_history'][-1]
values, dates = get_prediction_values_and_dates(last_snapshot)
p_idx    = pd.to_datetime(dates, utc=True)
pred     = pd.Series(values, index=p_idx, dtype="float64").sort_index()
bgi_pred = pred.shift(-1) - pred

df_b = df_day.copy()
if df_b.index.tz is None:
    df_b.index = df_b.index.tz_localize("UTC")
df_b['BGI'] = bgi_pred.reindex(df_b.index, method='nearest')
df_b = add_avg_delta_to_history_df(df_b)
df_b = add_deviation_to_history_df(df_b)

# --- Print comparison ---
print("\n" + "="*60)
print("  METHOD A — activity-based BGI (merged doses)")
print("="*60)
print(f"  avgDelta  mean : {df_a['avgDelta'].mean():.4f}")
print(f"  BGI       mean : {df_a['BGI'].mean():.4f}")
print(f"  deviation mean : {df_a['deviation'].mean():.4f}  ← should be ~0")
print(f"  deviation < 0  : {(df_a['deviation'] < 0).sum()} points")
print(f"  deviation > 0  : {(df_a['deviation'] > 0).sum()} points")

print("\n" + "="*60)
print("  METHOD B — prediction-based BGI (Loop's own output)")
print("="*60)
print(f"  avgDelta  mean : {df_b['avgDelta'].mean():.4f}")
print(f"  BGI       mean : {df_b['BGI'].mean():.4f}")
print(f"  deviation mean : {df_b['deviation'].mean():.4f}  ← should be ~0")
print(f"  deviation < 0  : {(df_b['deviation'] < 0).sum()} points")
print(f"  deviation > 0  : {(df_b['deviation'] > 0).sum()} points")

print("\n=== Sample rows: first 20 valid ===")
compare = pd.DataFrame({
    'CGM':      df_a['CGM'],
    'avgDelta': df_a['avgDelta'],
    'BGI_A':    df_a['BGI'],
    'dev_A':    df_a['deviation'],
    'BGI_B':    df_b['BGI'],
    'dev_B':    df_b['deviation'],
}).dropna().head(20)
print(compare.to_string())

compare.to_csv(os.path.join(SAVE_PATH, "deviation_comparison.csv"))
print(f"\n  Saved to {SAVE_PATH}/deviation_comparison.csv")