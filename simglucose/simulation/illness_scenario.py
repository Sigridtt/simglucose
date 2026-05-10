from datetime import timedelta
from collections import namedtuple
from .scenario import Scenario, parseTime

IllnessAction = namedtuple(
    "illness_scenario_action",
    ["meal", "glucose_offset", "reduction_factor", "rat_multiplier"],
)
IllnessAction.__new__.__defaults__ = (0, 0.0, 1.0, 1.0)


class IllnessScenario(Scenario):
    def __init__(
        self,
        start_time,
        meal_schedule=None,
        illness_start_step=288 * 4,
        illness_duration_steps=288 * 6,
        target_reduction_factor=0.60,
        target_rat_multiplier=1.30,
        max_glucose_offset_mmol_per_l=2.0,
        ramp_fraction=0.2,
    ):
        super().__init__(start_time=start_time)
        self.meal_schedule = meal_schedule or []
        self.illness_start_step = int(illness_start_step)
        self.illness_duration_steps = int(illness_duration_steps)
        self.illness_end_step = self.illness_start_step + self.illness_duration_steps
        self.target_reduction_factor = float(target_reduction_factor)
        self.target_rat_multiplier = float(target_rat_multiplier)
        self.max_glucose_offset_mgdl = float(max_glucose_offset_mmol_per_l) * 18.0

        self.ramp_up_steps = int(round(ramp_fraction * self.illness_duration_steps))
        self.ramp_down_steps = int(round(ramp_fraction * self.illness_duration_steps))
        self.steady_steps = self.illness_duration_steps - self.ramp_up_steps - self.ramp_down_steps

        self.reduction_factor_log = []

    def _minutes_since_start(self, t):
        delta_min = (t - self.start_time).total_seconds() / 60.0
        return int(round(delta_min))

    def _step_index_5min(self, t):
        return int(round(self._minutes_since_start(t) / 5.0))

    def _meal_amount(self, t):
        if not self.meal_schedule:
            return 0

        times, amounts = tuple(zip(*self.meal_schedule))
        parsed_times = [parseTime(item, self.start_time) for item in times]
        if t in parsed_times:
            idx = parsed_times.index(t)
            return amounts[idx]
        return 0

    def _illness_intensity(self, step_idx):
        if step_idx < self.illness_start_step or step_idx >= self.illness_end_step:
            return 0.0

        rel = step_idx - self.illness_start_step

        if self.ramp_up_steps > 0 and rel < self.ramp_up_steps:
            return rel / float(self.ramp_up_steps)

        plateau_start = self.ramp_up_steps
        plateau_end = self.ramp_up_steps + self.steady_steps

        if rel < plateau_end:
            return 1.0

        if self.ramp_down_steps <= 0:
            return 0.0

        down_rel = rel - plateau_end
        return max(0.0, 1.0 - down_rel / float(self.ramp_down_steps))

    def get_action(self, t):
        meal = self._meal_amount(t)
        step_idx = self._step_index_5min(t)
        intensity = self._illness_intensity(step_idx)

        reduction_factor = 1.0 - (1.0 - self.target_reduction_factor) * intensity
        rat_multiplier = 1.0 + (self.target_rat_multiplier - 1.0) * intensity

        self.reduction_factor_log.append(reduction_factor)

        return IllnessAction(
            meal=meal,
            glucose_offset=0.0,  # disabled
            reduction_factor=reduction_factor,
            rat_multiplier=rat_multiplier,
        )

    def reset(self):
        self.reduction_factor_log = []
