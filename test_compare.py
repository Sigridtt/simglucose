"""
Test: correct vs deliberately wrong ISF, for both LoopController and AdaptiveLoopController.
Patient: adult#001, 7 days, 3 meals/day.

Four conditions:
  1. LoopController       + correct ISF  (baseline)
  2. AdaptiveLoopController + correct ISF
  3. LoopController       + ISF * 1.5   (too high → under-dosing → high BG)
  4. AdaptiveLoopController + ISF * 1.5  (should correct toward true ISF)
"""

from datetime import datetime, timedelta
import os
import sys
import time
import pandas as pd

sys.dont_write_bytecode = True

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.analysis.report import report

from simglucose.controller.loop_ctrller import LoopController
from simglucose.controller.adaptive_loop_ctrller import AdaptiveLoopController

# ── Config ──────────────────────────────────────────────────────────────────
START_TIME   = datetime(2018, 1, 1, 8, 0, 0)
PATIENT_NAME = "adult#001"
DURATION     = timedelta(days=7)
ISF_MULTIPLIER = 1.2   # wrong ISF = true ISF * 1.2  (too high → under-dosing)

SAVE_PATH_BASE = os.path.join(
    os.path.dirname(__file__), "examples", "results", "test_compare_wrong_isf"
)

# 7 days of meals (3/day)
meals = []
for day in range(7):
    offset = day * 24
    meals.append((offset + 1,  45))
    meals.append((offset + 5,  70))
    meals.append((offset + 11, 60))

# ── Helper ───────────────────────────────────────────────────────────────────
def run_condition(label, controller, save_subdir):
    """Run one simulation condition and return the results DataFrame."""
    save_path = os.path.join(SAVE_PATH_BASE, save_subdir)
    os.makedirs(save_path, exist_ok=True)

    scenario   = CustomScenario(start_time=START_TIME, scenario=meals)
    patient    = T1DPatient.withName(PATIENT_NAME)
    sensor     = CGMSensor.withName("GuardianRT", seed=1)
    pump       = InsulinPump.withName("Insulet")
    env        = T1DSimEnv(patient, sensor, pump, scenario)
    sim_obj    = SimObj(env, controller, DURATION, animate=False, path=save_path)

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")
    result_df = sim(sim_obj)
    elapsed   = time.time() - t0
    print(f"  Done in {elapsed:.1f} s")

    all_df = pd.concat([result_df], keys=[PATIENT_NAME])
    results, _, _, _, _ = report(all_df, sensor, save_path)
    return results


def make_loop_ctrl():
    return LoopController(
        target=110,
        recommendation_type="automaticBolus",
        use_tdd_settings=True,
        insulin_type="novolog",
    )


def make_adaptive_ctrl():
    return AdaptiveLoopController(
        target=110,
        recommendation_type="automaticBolus",
        use_tdd_settings=True,
        insulin_type="novolog",
        warmup_days=1,
        adaptation_interval_hours=24,
        n_autotune_iterations=1,
    )


# ── Monkey-patch helper: override ISF after controller is built ───────────────
def force_wrong_isf(controller, multiplier):
    """
    Wrap get_therapy_settings_from_tdd so it returns ISF * multiplier.
    This simulates a misconfigured pump ISF without changing the patient model.
    """
    original_fn = controller.get_therapy_settings_from_tdd

    def patched(TDD):
        basal, isf, cr = original_fn(TDD)
        return basal, isf * multiplier, cr

    controller.get_therapy_settings_from_tdd = patched
    return controller


# ── Run all four conditions ───────────────────────────────────────────────────
print("Simulation starts ...")

# 1. LoopController + correct ISF
ctrl1 = make_loop_ctrl()
res1  = run_condition("LoopController — correct ISF", ctrl1, "loop_correct")

# 2. AdaptiveLoopController + correct ISF
ctrl2 = make_adaptive_ctrl()
res2  = run_condition("AdaptiveLoopController — correct ISF", ctrl2, "adaptive_correct")
isf_hist2 = ctrl2.get_isf_history(PATIENT_NAME)

# 3. LoopController + wrong ISF (1.5×)
ctrl3 = force_wrong_isf(make_loop_ctrl(), ISF_MULTIPLIER)
res3  = run_condition(f"LoopController — wrong ISF ({ISF_MULTIPLIER}×)", ctrl3, "loop_wrong_isf")

# 4. AdaptiveLoopController + wrong ISF (1.5×)
ctrl4 = force_wrong_isf(make_adaptive_ctrl(), ISF_MULTIPLIER)
res4  = run_condition(f"AdaptiveLoopController — wrong ISF ({ISF_MULTIPLIER}×)", ctrl4, "adaptive_wrong_isf")
isf_hist4 = ctrl4.get_isf_history(PATIENT_NAME)

# ── Print results table ───────────────────────────────────────────────────────
def row(label, res):
    return (
        f"  {label:<45}"
        f"  TIR={res['70<=BG<=180'].values[0]:5.1f}%"
        f"  TAR={res['BG>180'].values[0]:5.1f}%"
        f"  TBR={res['BG<70'].values[0]:5.1f}%"
        f"  Risk={res['Risk Index'].values[0]:.4f}"
    )

print("\n\n" + "="*80)
print("  RESULTS SUMMARY — adult#001, 7 days")
print("="*80)
print(row("LoopController — correct ISF",                        res1))
print(row("AdaptiveLoopController — correct ISF",               res2))
print(row(f"LoopController — wrong ISF ({ISF_MULTIPLIER}×)",    res3))
print(row(f"AdaptiveLoopController — wrong ISF ({ISF_MULTIPLIER}×)", res4))
print("="*80)

# ── Print ISF adaptation histories ───────────────────────────────────────────
print("\n  ISF history — AdaptiveLoopController, correct ISF:")
if isf_hist2:
    for e in isf_hist2:
        print(f"    {e['datetime']}  {e['old_isf']:.3f} → {e['new_isf']:.3f} mg/dL/U")
else:
    print("    (no adaptations)")

print(f"\n  ISF history — AdaptiveLoopController, wrong ISF ({ISF_MULTIPLIER}×):")
if isf_hist4:
    for e in isf_hist4:
        print(f"    {e['datetime']}  {e['old_isf']:.3f} → {e['new_isf']:.3f} mg/dL/U")
else:
    print("    (no adaptations)")

# ── Save summary ──────────────────────────────────────────────────────────────
os.makedirs(SAVE_PATH_BASE, exist_ok=True)
summary_lines = [
    "RESULTS SUMMARY — adult#001, 7 days",
    "="*80,
    row("LoopController — correct ISF",                        res1),
    row("AdaptiveLoopController — correct ISF",               res2),
    row(f"LoopController — wrong ISF ({ISF_MULTIPLIER}×)",    res3),
    row(f"AdaptiveLoopController — wrong ISF ({ISF_MULTIPLIER}×)", res4),
    "="*80,
    "",
    "ISF history — AdaptiveLoopController, correct ISF:",
]
for e in isf_hist2:
    summary_lines.append(f"  {e['datetime']}  {e['old_isf']:.3f} → {e['new_isf']:.3f}")
summary_lines.append("")
summary_lines.append(f"ISF history — AdaptiveLoopController, wrong ISF ({ISF_MULTIPLIER}×):")
for e in isf_hist4:
    summary_lines.append(f"  {e['datetime']}  {e['old_isf']:.3f} → {e['new_isf']:.3f}")

with open(os.path.join(SAVE_PATH_BASE, "summary.txt"), "w") as f:
    f.write("\n".join(summary_lines) + "\n")

print(f"\n  Summary saved to: {os.path.join(SAVE_PATH_BASE, 'summary.txt')}")