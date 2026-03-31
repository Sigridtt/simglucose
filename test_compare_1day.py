from datetime import datetime, timedelta
from simglucose.simulation.user_interface import simulate
from simglucose.controller.loop_ctrller import LoopController
from simglucose.controller.adaptive_loop_ctrller import AdaptiveLoopController
from simglucose.simulation.scenario import CustomScenario
import os
import sys
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s - %(message)s'
)

sys.dont_write_bytecode = True

START_TIME = datetime(2018, 1, 1, 8, 0, 0)
PATIENT_NAME = "adult#010"

# Keep scenario identical for both controllers 
SCENARIO = CustomScenario(
    start_time=START_TIME,
    scenario=[(1, 45), (5, 70), (11, 60)],
)

BASE_SAVE_PATH = os.path.join(
    os.path.dirname(__file__),
    "examples",
    "results",
    "compare_loop",
    "#adult010_1day"
)
os.makedirs(BASE_SAVE_PATH, exist_ok=True)


def run_loop():
    controller = LoopController(
        target=110,
        recommendation_type='automaticBolus',
        use_tdd_settings=True,
        insulin_type='novolog',
    )

    save_path = os.path.join(BASE_SAVE_PATH, "loop")
    os.makedirs(save_path, exist_ok=True)

    results = simulate(
        sim_time=timedelta(hours=24),
        scenario=SCENARIO,
        controller=controller,
        patient_names=[PATIENT_NAME],
        cgm_name="GuardianRT",
        cgm_seed=1,
        insulin_pump_name="Insulet",
        start_time=START_TIME,
        save_path=save_path,
        animate=False,
        parallel=False,
    )
    return results


def run_adaptive():
    controller = AdaptiveLoopController(
        target=110,
        recommendation_type='automaticBolus',
        use_tdd_settings=True,
        insulin_type='novolog',
        warmup_days=1,
        adaptation_interval_hours=24,
        n_autotune_iterations=1,
        debug_timing=False,
    )

    save_path = os.path.join(BASE_SAVE_PATH, "adaptive")
    os.makedirs(save_path, exist_ok=True)

    results = simulate(
        sim_time=timedelta(hours=24),
        scenario=SCENARIO,
        controller=controller,
        patient_names=[PATIENT_NAME],
        cgm_name="GuardianRT",
        cgm_seed=1,
        insulin_pump_name="Insulet",
        start_time=START_TIME,
        save_path=save_path,
        animate=False,
        parallel=False,
    )
    return results


def run_csv_path(controller_name):
    return os.path.join(BASE_SAVE_PATH, controller_name, f"{PATIENT_NAME}.csv")


def extract_run_stats(controller_name):
    csv_path = run_csv_path(controller_name)
    df = pd.read_csv(csv_path)
    total_insulin = float(df['insulin'].sum())
    max_bg = float(df['BG'].max())
    return total_insulin, max_bg


def metric_line(label, results_df):
    return (
        f"{label:<12} "
        f"TIR={results_df['70<=BG<=180'].values[0]:6.2f}%  "
        f"TAR={results_df['BG>180'].values[0]:6.2f}%  "
        f"TBR={results_df['BG<70'].values[0]:6.2f}%  "
        f"Risk Index={results_df['Risk Index'].values[0]:.4f}"
    )


print("Simulation starts ...")
loop_results = run_loop()
adaptive_results = run_adaptive()
loop_total_insulin, loop_max_bg = extract_run_stats("loop")
adaptive_total_insulin, adaptive_max_bg = extract_run_stats("adaptive")

print("\n=== 1-day comparison (same scenario, same seed) ===")
print(metric_line("Loop", loop_results))
print(metric_line("Adaptive", adaptive_results))
print(f"Loop         Total insulin={loop_total_insulin:.4f} U  Max BG={loop_max_bg:.2f} mg/dL")
print(f"Adaptive     Total insulin={adaptive_total_insulin:.4f} U  Max BG={adaptive_max_bg:.2f} mg/dL")

summary_lines = [
    "1-day comparison: Loop vs Adaptive (same scenario, same seed)",
    "=" * 80,
    metric_line("Loop", loop_results),
    metric_line("Adaptive", adaptive_results),
    "",
    "Delta (Adaptive - Loop):",
    (
        f"TIR={adaptive_results['70<=BG<=180'].values[0] - loop_results['70<=BG<=180'].values[0]:+.2f}%  "
        f"TAR={adaptive_results['BG>180'].values[0] - loop_results['BG>180'].values[0]:+.2f}%  "
        f"TBR={adaptive_results['BG<70'].values[0] - loop_results['BG<70'].values[0]:+.2f}%  "
        f"Risk Index={adaptive_results['Risk Index'].values[0] - loop_results['Risk Index'].values[0]:+.4f}"
    ),
    (
        f"Total insulin={adaptive_total_insulin - loop_total_insulin:+.4f} U  "
        f"Max BG={adaptive_max_bg - loop_max_bg:+.2f} mg/dL"
    ),
    "",
    "Run-level stats:",
    f"Loop:     Total insulin={loop_total_insulin:.4f} U  Max BG={loop_max_bg:.2f} mg/dL",
    f"Adaptive: Total insulin={adaptive_total_insulin:.4f} U  Max BG={adaptive_max_bg:.2f} mg/dL",
    "",
    "Full Loop metrics:",
    pd.DataFrame(loop_results).to_string(),
    "",
    "Full Adaptive metrics:",
    pd.DataFrame(adaptive_results).to_string(),
    "",
    "Saved result folders:",
    f"- Loop: {os.path.join(BASE_SAVE_PATH, 'loop')}",
    f"- Adaptive: {os.path.join(BASE_SAVE_PATH, 'adaptive')}",
]

summary_path = os.path.join(BASE_SAVE_PATH, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")

print(f"\nSummary saved to: {summary_path}")
