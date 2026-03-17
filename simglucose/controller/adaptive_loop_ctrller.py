from .loop_ctrller import LoopController
from .base import Action

from loop_to_python_api.helpers import get_json_loop_prediction_input_from_df
import loop_to_python_api.api as loop_to_python_api

from loop_to_python_adaptive.autotune_isf import (
    run_autotune_isf_iterations,
    AutotuneISFConfig,
)

import numpy as np
import pandas as pd
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class AdaptiveLoopController(LoopController):
    """
    Extends LoopController with oref0-style ISF autotune.

    Key fix: instead of keeping only the last json_data, we accumulate
    ALL json inputs from the adaptation window and merge their dose lists
    before passing to autotune. This ensures BGI computation has the full
    picture of insulin delivered throughout the day, not just the last 12h.
    """

    def __init__(
        self,
        target=110,
        recommendation_type='automaticBolus',
        use_tdd_settings=True,
        use_fully_closed_loop=False,
        insulin_type='novolog',
        warmup_days=1,
        adaptation_interval_hours=24,
        n_autotune_iterations=1,
        autotune_cfg=None,
    ):
        self.max_window_days = 3  # grow to 3 days then roll
        super().__init__(
            target=target,
            recommendation_type=recommendation_type,
            use_tdd_settings=use_tdd_settings,
            use_fully_closed_loop=use_fully_closed_loop,
            insulin_type=insulin_type,
        )
        self.warmup_duration = timedelta(days=warmup_days)
        self.adaptation_interval = timedelta(hours=adaptation_interval_hours)
        self.n_autotune_iterations = n_autotune_iterations
        self.autotune_cfg = autotune_cfg or AutotuneISFConfig()
        self._adaptive_state = {}

    # ------------------------------------------------------------------
    #   Core policy override
    # ------------------------------------------------------------------

    def _loop_policy(self, datetime, name, meal, glucose, env_sample_time, TDD=None):

        # --- Resolve patient parameters ---
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
            basal = basal_pr_hr / 60
        else:
            basal = u2ss * BW / 6000
            basal_pr_hr = basal * 60
            cr = float(quest.CR.values[0])
            isf_pump = float(quest.CF.values[0])

        # --- Initialise adaptive state on first call ---
        if name not in self._adaptive_state:
            self._adaptive_state[name] = {
                'isf':          isf_pump,
                'isf_pump':     isf_pump,
                'sim_start':    datetime,
                'last_adapted': datetime,
                'log_rows':     [],
                'json_history': [],   # ALL json inputs this window
                'isf_history':  [],
            }

        state = self._adaptive_state[name]
        isf = state['isf']

        meal_grams = meal * env_sample_time
        if self.use_fully_closed_loop:
            meal_grams = 0
            meal = 0

        # --- Observation bookkeeping ---
        df_observations = self.add_patient_observation(
            name, datetime, glucose, np.nan, np.nan, meal_grams, TDD
        )

        # Loop warmup: < 3 hours of data
        if len(df_observations) < (3 * 60 // env_sample_time):
            self.add_patient_observation(
                name, datetime, glucose, basal_pr_hr, 0, meal_grams, TDD
            )
            self._log_row(state, datetime, glucose, basal_pr_hr, 0, meal_grams)
            return Action(basal=basal, bolus=0)

        # --- Build Loop algorithm input ---
        df_tail = df_observations.sort_index().tail(int(12 * 60 // env_sample_time))
        json_data = get_json_loop_prediction_input_from_df(
            df_tail, basal_pr_hr, isf, cr,
            prediction_start=datetime,
            insulin_type=self.insulin_type,
        )
        json_data['maxBasalRate'] = basal_pr_hr * 2
        json_data['recommendationType'] = self.recommendation_type

        # Accumulate json inputs — MIGHT CAUSE MEMORY BLOAT IF SIMULATION IS VERY LONG WITHOUT ADAPTATION, SO WE CLEAR IT AFTER ADAPTATION
        state['json_history'].append(json_data)

        # --- Get dose recommendation ---
        dose_recommendations = loop_to_python_api.get_dose_recommendations(json_data)
        basal_rec = dose_recommendations['automatic']['basalAdjustment']['unitsPerHour']

        if meal > 0:
            json_data['recommendationType'] = 'manualBolus'
            dose_recommendations = loop_to_python_api.get_dose_recommendations(json_data)
            bolus_rec = dose_recommendations['manual']['amount']
        elif 'bolusUnits' in dose_recommendations['automatic']:
            bolus_rec = dose_recommendations['automatic']['bolusUnits']
        else:
            bolus_rec = 0.0

        self.add_patient_observation(
            name, datetime, glucose,
            basal=basal_rec, bolus=bolus_rec, carbs=meal_grams, TDD=TDD
        )

        self._log_row(state, datetime, glucose, basal_rec, bolus_rec, meal_grams)
        self._maybe_adapt(name, datetime)

        return Action(basal=basal_rec / 60, bolus=bolus_rec / env_sample_time)

    # ------------------------------------------------------------------
    #   Logging
    # ------------------------------------------------------------------

    def _log_row(self, state, datetime, cgm, basal, bolus, carbs):
        state['log_rows'].append({
            'date':  datetime,
            'CGM':   cgm,
            'basal': basal,
            'bolus': bolus,
            'carbs': np.nan if carbs <= 0 else carbs,
        })

    def _build_daily_log_df(self, log_rows):
        df = pd.DataFrame(log_rows).set_index('date')
        df.index.name = 'date'
        return df

    def _merge_json_inputs(self, json_history):
        """
        Merge dose lists from all accumulated json inputs into one.
        Uses the most recent json as the base (it has the latest profile
        settings), then adds any unique doses from earlier snapshots.
        This gives BGI computation the full day's insulin history.
        """
        if not json_history:
            return None

        # Start from the most recent snapshot
 
        merged = dict(json_history[-1])
        merged['doses'] = list(merged.get('doses', []) or [])

        # Merge doses
        seen_starts = {d.get('startDate') for d in merged['doses']}
        for json_input in reversed(json_history[:-1]):
            for dose in (json_input.get('doses') or []):
                start = dose.get('startDate')
                if start and start not in seen_starts:
                    merged['doses'].append(dose)
                    seen_starts.add(start)
        merged['doses'].sort(key=lambda d: d.get('startDate', ''))

        # Merge carbEntries — collect all non-zero entries across all snapshots
        seen_carb_dates = set()
        all_carb_entries = []
        for json_input in json_history:
            for entry in (json_input.get('carbEntries') or []):
                if float(entry.get('grams', 0)) > 0:
                    date = entry.get('date')
                    if date and date not in seen_carb_dates:
                        all_carb_entries.append(entry)
                        seen_carb_dates.add(date)
        all_carb_entries.sort(key=lambda e: e.get('date', ''))
        merged['carbEntries'] = all_carb_entries

        return merged
    # ------------------------------------------------------------------
    #   Autotune trigger
    # ------------------------------------------------------------------

    def _maybe_adapt(self, name, datetime):
        state = self._adaptive_state[name]

        elapsed = datetime - state['sim_start']
    
        if elapsed < self.warmup_duration:
            return
        
        if datetime - state['last_adapted'] < self.adaptation_interval:
            return

        if not state['json_history'] or len(state['log_rows']) < 12:
            return

        print(
            f"\n  [Autotune] {name} @ {datetime} — running ISF autotune "
            f"({len(state['log_rows'])} rows, {len(state['json_history'])} json snapshots)..."
        )

        try:
            daily_log_df  = self._build_daily_log_df(state['log_rows'])
            merged_json   = self._merge_json_inputs(state['json_history'])
            snapshot = state['json_history'][0]
            print(f"  keys in snapshot: {list(snapshot.keys())}")
            print(f"  carbEntries in snapshot: {len(snapshot.get('carbEntries', []))}")

            result = run_autotune_isf_iterations(
                df_windows=[daily_log_df],
                loop_algorithm_inputs=[merged_json],
                pump_isf=state['isf_pump'],   # ← original pump ISF, fixed forever
                isf_current=state['isf'],     # ← yesterday's tuned ISF
                n_iterations=self.n_autotune_iterations,
                cfg=self.autotune_cfg,
                json_history_list=[state['json_history']],
            )
            new_isf = result['finalISF']
            old_isf = state['isf']

            state['isf'] = new_isf
            state['isf_history'].append({
                'datetime': datetime,
                'old_isf':  old_isf,
                'new_isf':  new_isf,
            })
            
            #Growing data up to 3 days, day 2 uses 1 day of data, day 3 uses 2 days, day 4 and out uses 3 days of data
            state['last_adapted'] = datetime

            # How many rows = 1 day's worth (288 = 24h * 60min / 5min)
            one_day_rows = int(24 * 60 / 5)

            while len(state['log_rows']) > self.max_window_days * one_day_rows:
                state['log_rows']     = state['log_rows'][one_day_rows:]
                state['json_history'] = state['json_history'][one_day_rows:]


            print(f"  [Autotune] {name}: ISF {old_isf:.3f} → {new_isf:.3f} mg/dL/U")

        except Exception as e:
            logger.warning(
                f"[{name}] Autotune failed: {e}. Keeping ISF={state['isf']:.3f}"
            )
            state['last_adapted'] = datetime
            state['log_rows']     = []
            state['json_history'] = []

    # ------------------------------------------------------------------
    #   Public helpers
    # ------------------------------------------------------------------

    def get_isf_history(self, patient_name):
        return self._adaptive_state.get(patient_name, {}).get('isf_history', [])

    def get_current_isf(self, patient_name):
        return self._adaptive_state.get(patient_name, {}).get('isf', None)

    def reset(self):
        self._adaptive_state = {}