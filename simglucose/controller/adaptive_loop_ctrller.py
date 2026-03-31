from .loop_ctrller import LoopController
from .base import Action
from loop_to_python_api.helpers import get_json_loop_prediction_input_from_df
import loop_to_python_api.api as loop_to_python_api
from loop_to_python_adaptive.adaptive_manager import AdaptiveManager
import numpy as np
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)


class AdaptiveLoopController(LoopController):
    """
    Two-layer adaptive controller (autotune + autosens).

    """

    def __init__(
        self,
        target=100, #oref0 default
        recommendation_type='automaticBolus',
        use_tdd_settings=True,
        use_fully_closed_loop=False,
        insulin_type='novolog',
        warmup_days=1,
        adaptation_interval_hours=24,
        n_autotune_iterations=1,
        autotune_cfg=None,
        autosens_cfg=None,
        debug_timing=False,
        debug_every_steps=12,
        slow_step_seconds=2.0,
    ):
        super().__init__(
            target=target,
            recommendation_type=recommendation_type,
            use_tdd_settings=use_tdd_settings,
            use_fully_closed_loop=use_fully_closed_loop,
            insulin_type=insulin_type,
        )
        self.manager = AdaptiveManager(
            warmup_days=warmup_days,
            adaptation_interval_hours=adaptation_interval_hours,
            n_autotune_iterations=n_autotune_iterations,
            autotune_cfg=autotune_cfg,
            autosens_cfg=autosens_cfg,
        )
        self.debug_timing = debug_timing
        self.debug_every_steps = max(1, int(debug_every_steps))
        self.slow_step_seconds = float(slow_step_seconds)
        self._debug_step_counts = {}

    def _loop_policy(self, datetime, name, meal, glucose, env_sample_time, TDD=None):
        step_count = self._debug_step_counts.get(name, 0) + 1
        self._debug_step_counts[name] = step_count
        should_log = self.debug_timing and (step_count % self.debug_every_steps == 0)
        t0 = time.perf_counter()
        t_prev = t0

        def _checkpoint(stage):
            nonlocal t_prev
            if not should_log:
                return
            now = time.perf_counter()
            logger.info(
                '[AdaptiveLoopController] patient=%s step=%s stage=%s dt=%.3fs total=%.3fs glucose=%.1f meal=%.3f',
                name, step_count, stage, now - t_prev, now - t0, float(glucose), float(meal)
            )
            t_prev = now

        _checkpoint('start')
        
        # --- Resolve patient parameters (same as base) ---
        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(name)]
            u2ss = params.u2ss.values.item()
            BW = params.BW.values.item()
            TDD = quest.TDI.values[0]
        else:
            quest = pd.DataFrame(
                [['Average', 1 / 15, 1 / 50, 50, 30]],
                columns=['Name', 'CR', 'CF', 'TDI', 'Age']
            )
            u2ss = 1.43
            BW = 57.0
            if TDD is None:
                TDD = 50

        if self.use_tdd_settings:
            basal_pr_hr, isf_pump, cr = self.get_therapy_settings_from_tdd(TDD)
        else:
            basal_pr_hr = u2ss * BW / 6000 * 60
            cr = float(quest.CR.values[0])
            isf_pump = float(quest.CF.values[0])
        _checkpoint('resolved_settings')

        meal_grams = meal * env_sample_time
        if self.use_fully_closed_loop:
            meal_grams = 0
            meal = 0

        # Initialize adaptive state if needed
        if name not in self.manager.patients:
            self.manager.initialize_patient(name, datetime, isf_pump, cr, basal_pr_hr)

        df_observations = self.add_patient_observation(
            name, datetime, glucose, np.nan, np.nan, meal_grams, TDD
        )
        _checkpoint('added_observation')

        # Warmup
        if len(df_observations) < (3 * 60 // env_sample_time):
            self.add_patient_observation(
                name, datetime, glucose, basal_pr_hr, 0, meal_grams, TDD
            )
            _checkpoint('warmup_return')
            return Action(basal=basal_pr_hr/60, bolus=0)

        # Build JSON for autotune layer
        df_tail = df_observations.sort_index().tail(int(12 * 60 // env_sample_time))
        json_autotune = get_json_loop_prediction_input_from_df(
            df_tail,
            self.manager.patients[name].basal_pr_hr,
            self.manager.patients[name].isf,
            self.manager.patients[name].cr,
            prediction_start=datetime,
            insulin_type=self.insulin_type,
        )
        json_autotune['maxBasalRate'] = self.manager.patients[name].basal_pr_hr * 2
        json_autotune['recommendationType'] = self.recommendation_type
        _checkpoint('prepared_autotune_json')

        # Run adaptive step (autosens + check autotune)
        _checkpoint('before_manage_step')
        effective_isf, effective_basal, effective_cr, effective_min_bg, effective_max_bg = \
            self.manager.manage_step(name, datetime, glucose, json_autotune)
        _checkpoint('after_manage_step')


        # Build JSON for dose with effective (two-layer) params
        json_dose = get_json_loop_prediction_input_from_df(
            df_tail,
            effective_basal,
            effective_isf,
            effective_cr,
            prediction_start=datetime,
            insulin_type=self.insulin_type,
        )
        json_dose['targets'] = {'low': int(effective_min_bg),'high': int(effective_max_bg),}
        json_dose['maxBasalRate'] = effective_basal * 2
        json_dose['recommendationType'] = self.recommendation_type
        _checkpoint('prepared_dose_json')

        # Get dose
        _checkpoint('before_dose_recommendation_auto')
        dose_recommendations = loop_to_python_api.get_dose_recommendations(json_dose)
        _checkpoint('after_dose_recommendation_auto')
        basal_rec = dose_recommendations['automatic']['basalAdjustment']['unitsPerHour']

        if meal > 0:
            json_dose['recommendationType'] = 'manualBolus'
            _checkpoint('before_dose_recommendation_manual')
            dose_recommendations = loop_to_python_api.get_dose_recommendations(json_dose)
            _checkpoint('after_dose_recommendation_manual')
            bolus_rec = dose_recommendations['manual']['amount']
        elif 'bolusUnits' in dose_recommendations['automatic']:
            bolus_rec = dose_recommendations['automatic']['bolusUnits']
        else:
            bolus_rec = 0.0

        self.add_patient_observation(
            name, datetime, glucose,
            basal=basal_rec, bolus=bolus_rec, carbs=meal_grams, TDD=TDD
        )
        _checkpoint('saved_action')

        if self.debug_timing:
            total_s = time.perf_counter() - t0
            if total_s >= self.slow_step_seconds:
                logger.warning(
                    '[AdaptiveLoopController] slow step patient=%s step=%s total=%.3fs',
                    name, step_count, total_s
                )

        return Action(basal=basal_rec / 60, bolus=bolus_rec / env_sample_time)

    # Public API to query adaptation results
    def get_isf_history(self, patient_name):
        return self.manager.get_isf_history(patient_name)

    def get_current_isf(self, patient_name):
        return self.manager.get_current_isf(patient_name)

    def get_cr_history(self, patient_name):
        return self.manager.get_cr_history(patient_name)

    def get_current_cr(self, patient_name):
        return self.manager.get_current_cr(patient_name)

    def get_basal_history(self, patient_name):
        return self.manager.get_basal_history(patient_name)

    def get_current_basal(self, patient_name):
        return self.manager.get_current_basal(patient_name)

    def get_autosens_log(self, patient_name):
        return self.manager.get_autosens_log(patient_name)

    def reset(self):
        super().reset()
        self.manager.reset()