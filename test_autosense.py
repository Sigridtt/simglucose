"""
Test autosens functionality.

What we verify:
  1. During fasting periods (no COB) the ratio moves away from 1.0
     when the pump ISF is wrong (1.3x too high).
  2. During meal periods (COB > 0) points are excluded and the ratio
     does not spike due to meal absorption noise.
  3. The ratio stays within [autosens_min, autosens_max] = [0.7, 1.2].
  4. The effective ISF differs from the autotune ISF when ratio != 1.0.
  5. With a correct ISF the ratio stays close to 1.0 throughout.
"""

from datetime import datetime, timedelta
import os, sys
import pandas as pd
import numpy as np

sys.dont_write_bytecode = True

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario

from simglucose.controller.adaptive_loop_ctrller import AdaptiveLoopController
from loop_to_python_adaptive.autosens_isf import AutosensConfig

START_TIME   = datetime(2018, 1, 1, 8, 0, 0)
PATIENT_NAME = "adult#001"
DURATION     = timedelta(days=3)   # short enough to run fast

# 3 days of meals
meals = []
for day in range(3):
    offset = day * 24
    meals.append((offset + 1,  45))
    meals.append((offset + 5,  70))
    meals.append((offset + 11, 60))

SAVE_BASE = os.path.join(os.path.dirname(__file__), "examples", "results", "test_autosens")
def run_condition(label, initial_isf_multiplier, save_sub):
    save_path = os.path.join(SAVE_BASE, save_sub)
    os.makedirs(save_path, exist_ok=True)

    ctrl = AdaptiveLoopController(
        target=110,
        recommendation_type='automaticBolus',
        use_tdd_settings=True,
        insulin_type='novolog',
        warmup_days=1,
        adaptation_interval_hours=24,
        autosens_cfg=AutosensConfig(
            autosens_max=1.2,
            autosens_min=0.7,
            min_points=10,
        ),
    )

    # ── Patch the initial ISF in adaptive state on first patient call ──
    # We intercept _loop_policy's first initialisation of state['isf']
    # by wrapping get_therapy_settings_from_tdd so the pump ISF is correct
    # (patient physiology unchanged) but the controller STARTS with a wrong ISF.
    # This simulates a clinician having programmed the wrong ISF initially.
    original_init = ctrl.__class__._loop_policy

    first_call = {'done': False}

    def patched_loop_policy(self, datetime, name, meal, glucose, env_sample_time, TDD=None):
        result = original_init(self, datetime, name, meal, glucose, env_sample_time, TDD)
        # On the very first call, after state is initialised, scale the ISF
        if not first_call['done'] and name in self._adaptive_state:
            state = self._adaptive_state[name]
            state['isf']      = state['isf_pump'] * initial_isf_multiplier
            first_call['done'] = True
        return result

    import types
    ctrl._loop_policy = types.MethodType(patched_loop_policy, ctrl)
    # ──────────────────────────────────────────────────────────────────

    scenario = CustomScenario(start_time=START_TIME, scenario=meals)
    patient  = T1DPatient.withName(PATIENT_NAME)
    sensor   = CGMSensor.withName("GuardianRT", seed=1)
    pump     = InsulinPump.withName("Insulet")
    env      = T1DSimEnv(patient, sensor, pump, scenario)
    sim_obj  = SimObj(env, ctrl, DURATION, animate=False, path=save_path)
    

    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")
    sim(sim_obj)

    log = ctrl.get_autosens_log(PATIENT_NAME)
    return pd.DataFrame(log), ctrl
# ── Run two conditions ────────────────────────────────────────────────────────
df_correct, ctrl_correct = run_condition(
    "Correct initial ISF (ratio should stay ~1.0)",
    initial_isf_multiplier=1.0,
    save_sub="correct_isf",
)

df_wrong, ctrl_wrong = run_condition(
    "Initial ISF too high x1.3 (autosens should compensate, autotune should correct)",
    initial_isf_multiplier=1.3,
    save_sub="wrong_initial_isf_1_3x",
)

# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("  AUTOSENS TEST RESULTS")
print("="*70)

for label, df in [("Correct ISF", df_correct), ("Wrong ISF x1.3", df_wrong)]:
    if df.empty:
        print(f"\n  {label}: no autosens log — check _loop_policy logging")
        continue

    ratio_col = df['ratio'].dropna()
    fasting    = df[df['cob'] == 0]['ratio'].dropna()
    meal       = df[df['cob'] >  0]['ratio'].dropna()

    print(f"\n  {label}:")
    print(f"    Total steps logged      : {len(df)}")
    print(f"    Steps with COB=0 (fasting): {len(fasting)}")
    print(f"    Steps with COB>0 (meal)  : {len(meal)}")
    print(f"    Ratio — overall  mean/std: {ratio_col.mean():.4f} / {ratio_col.std():.4f}")
    print(f"    Ratio — fasting  mean/std: {fasting.mean():.4f} / {fasting.std():.4f}" if len(fasting) else "    Ratio — fasting: no data")
    print(f"    Ratio — meal     mean/std: {meal.mean():.4f} / {meal.std():.4f}" if len(meal) else "    Ratio — meal: no data")
    print(f"    Ratio min/max            : {ratio_col.min():.4f} / {ratio_col.max():.4f}")
    print(f"    Ratio always in [0.7,1.2]: {(ratio_col >= 0.7).all() and (ratio_col <= 1.2).all()}")

    # Verify meal periods are excluded (ratio should not spike during meals)
    if len(meal) > 0 and len(fasting) > 0:
        meal_vs_fasting = abs(meal.mean() - 1.0) <= abs(fasting.mean() - 1.0) + 0.05
        print(f"    Meal ratio not more extreme than fasting: {meal_vs_fasting}")

# ── Save logs ────────────────────────────────────────────────────────────────
df_correct.to_csv(os.path.join(SAVE_BASE, "autosens_log_correct.csv"), index=False)
df_wrong.to_csv(os.path.join(SAVE_BASE,   "autosens_log_wrong.csv"),   index=False)
print(f"\n  Logs saved to: {SAVE_BASE}")

# ── What to look for ─────────────────────────────────────────────────────────
print("""
  WHAT TO LOOK FOR:
  -----------------
  Correct ISF:
    - Ratio should stay close to 1.0 (mean ~0.95-1.05)
    - Small fluctuations are normal (noise in CGM and BGI)

  Wrong ISF x1.3:
    - Pump ISF is 30% too high → under-dosing → BG runs higher than expected
    - Deviations will be systematically positive during fasting
    - Autosens ratio should drift below 1.0 (toward ~0.77 = 1/1.3)
      meaning: "patient appears more sensitive than pump thinks"
    - effective_isf = autotune_isf / ratio → effective ISF goes DOWN
      → Loop gives more insulin to compensate

  If wrong ISF ratio stays at 1.0:
    - Check that _get_latest_bgi returns non-zero values
    - Check that _get_latest_deviation returns non-zero values
    - Check that COB is correctly excluding meal periods (not too aggressive)
    - Print df_wrong[['bgi','deviation','cob','ratio']].head(50) to inspect
""")

# ── Diagnose why ratio is stuck at 1.0 ───────────────────────────────────────
print("\n\n  DIAGNOSIS — inspecting autosens buffer content")
print("  " + "-"*50)

for label, df in [("Correct ISF", df_correct), ("Wrong ISF x1.3", df_wrong)]:
    print(f"\n  {label} — first 20 rows of bgi/deviation/cob:")
    print(df[['datetime','bgi','deviation','cob','ratio']].head(20).to_string(index=False))
    print(f"\n  {label} — non-zero bgi count    : {(df['bgi'] != 0).sum()} / {len(df)}")
    print(f"  {label} — non-zero deviation    : {(df['deviation'] != 0).sum()} / {len(df)}")
    print(f"  {label} — zero cob count        : {(df['cob'] == 0).sum()} / {len(df)}")
    print(f"  {label} — small |bgi| (<1e-6)   : {(df['bgi'].abs() < 1e-6).sum()} / {len(df)}")
    print(f"  {label} — large |dev| (>6)      : {(df['deviation'].abs() > 6).sum()} / {len(df)}")