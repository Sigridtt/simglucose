from .loop_ctrller import LoopController
from .base import Action
from loop_to_python_api.helpers import get_json_loop_prediction_input_from_df
import loop_to_python_api.api as loop_to_python_api
from loop_to_python_adaptive.adaptive_manager import AdaptiveManager
from loop_to_python_adaptive.autosens import AutosensConfig
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
        target=100,
        recommendation_type="automaticBolus",
        use_tdd_settings=True,
        use_fully_closed_loop=False,
        insulin_type="novolog",
        warmup_days=1, #warmup autosens, autotune has 7 days
        autotune_interval_hours=24,
        autotune_cfg=None,
        autosens_cfg=None,
        autosens_max=1.2,  
        autosens_min=0.7,
        debug_timing=False,
        debug_every_steps=12,
        slow_step_seconds=2.0,
        # ── ablation flags ──────────────────────────
        enable_autotune: bool = True,
        enable_autosens: bool = True,
        enable_sick_detection: bool = True,
    ):
        super().__init__(
            target=target,
            recommendation_type=recommendation_type,
            use_tdd_settings=use_tdd_settings,
            use_fully_closed_loop=use_fully_closed_loop,
            insulin_type=insulin_type,
        )
        if autosens_cfg is None:
            autosens_cfg = AutosensConfig(
                autosens_max=autosens_max,
                autosens_min=autosens_min,
            )
        self.manager = AdaptiveManager(
            warmup_days=warmup_days,
            autotune_interval_hours=autotune_interval_hours,
            autotune_cfg=autotune_cfg,
            autosens_cfg=autosens_cfg,
            enable_autotune=enable_autotune,
            enable_autosens=enable_autosens,
            enable_sick_detection=enable_sick_detection,
        )

        self.debug_timing = debug_timing
        self.debug_every_steps = max(1, int(debug_every_steps))
        self.slow_step_seconds = float(slow_step_seconds)
        self._debug_step_counts = {}
        self._last_basal = {}

    # ----------------------------
    # CORE FIX: defensive typing
    # ----------------------------
    def _safe_str(self, x):
        return "" if x is None else str(x)

    def _safe_float(self, x, default=0.0):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return default
            return float(x)
        except Exception:
            return default

        
    def _clamp_dose_inputs(self, json_dose):
        def safe(x, fallback):
            try:
                if x is None:
                    return fallback
                if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
                    return fallback
                return float(x)
            except Exception:
                return fallback
        # basal safety
        if "maxBasalRate" in json_dose:
            json_dose["maxBasalRate"] = max(0.05, min(10.0, safe(json_dose["maxBasalRate"], 1.0)))

        # ensure target valid
        if "target" in json_dose:
            low = safe(json_dose["target"].get("low", 90), 90)
            high = safe(json_dose["target"].get("high", 150), 150)

            if low >= high:
                low, high = 80, 180

            json_dose["target"]["low"] = int(low)
            json_dose["target"]["high"] = int(high)

        return json_dose

    def _loop_policy(self, datetime, name, meal, glucose, env_sample_time, TDD=None):

        logger = logging.getLogger(__name__)

        # -----------------------------
        # SAFE CASTING
        # -----------------------------
        name = "" if name is None else str(name)

        def safe_float(x, default=0.0):
            try:
                if x is None:
                    return default
                x = float(x)
                if np.isnan(x) or np.isinf(x):
                    return default
                return x
            except Exception:
                return default

        meal = safe_float(meal, 0.0)
        glucose = safe_float(glucose, 100.0)
        env_sample_time = safe_float(env_sample_time, 5.0)

        if datetime is None or pd.isna(datetime):
            datetime = pd.Timestamp.utcnow()
        datetime = pd.Timestamp(datetime)

        # -----------------------------
        # PATIENT RESOLUTION
        # -----------------------------
        if name in self.quest["Name"].astype(str).values:
            quest = self.quest[self.quest["Name"].astype(str) == name]
            params = self.patient_params[self.patient_params["Name"].astype(str) == name]

            u2ss = float(params.u2ss.values[0])
            BW = float(params.BW.values[0])
            TDD = float(quest.TDI.values[0])
        else:
            quest = pd.DataFrame(
                [["Average", 1 / 15, 1 / 50, 50, 30]],
                columns=["Name", "CR", "CF", "TDI", "Age"],
            )
            u2ss = 1.43
            BW = 57.0
            TDD = 50 if TDD is None else float(TDD)

        # -----------------------------
        # THERAPY SETTINGS
        # -----------------------------
        if self.use_tdd_settings:
            basal_pump, isf_pump, cr = self.get_therapy_settings_from_tdd(TDD)
        else:
            basal_pump = u2ss * BW / 6000 * 60
            cr = float(quest.CR.values[0])
            isf_pump = float(quest.CF.values[0])

        meal_grams = meal * env_sample_time if not self.use_fully_closed_loop else 0.0

        if name not in self.manager.patients:
            self.manager.initialize_patient(name, datetime, isf_pump, cr, basal_pump)

        df_observations = self.add_patient_observation(
            name, datetime, glucose, np.nan, np.nan, meal_grams, TDD
        )

        if len(df_observations) < (3 * 60 // env_sample_time):
            if glucose < 80:
                return Action(basal=0.0, bolus=0.0)
            return Action(basal=basal_pump / 60, bolus=0.0)

        df_tail = df_observations.sort_index().tail(int(12 * 60 // env_sample_time))

        # -----------------------------
        # BUILD AUTOTUNE JSON
        # -----------------------------
        json_dose = get_json_loop_prediction_input_from_df(
            df_tail,
            self.manager.patients[name].basal,
            self.manager.patients[name].isf,
            self.manager.patients[name].cr,
            prediction_start=datetime,
            insulin_type=self.insulin_type,
        )

        # -----------------------------
        # HARD SANITIZATION (CRITICAL)
        # -----------------------------
        def sanitize(x):
            if isinstance(x, (np.ndarray, list, tuple)):
                x = np.asarray(x).flatten()[0]
            try:
                x = float(x)
            except Exception:
                return 0.0
            if np.isnan(x) or np.isinf(x):
                return 0.0
            return x

        # -----------------------------
        # ADAPTIVE STEP
        # -----------------------------
        eff_isf, eff_basal, eff_cr, eff_min_bg, eff_max_bg = self.manager.manage_step(
            name, datetime, glucose, json_dose
        )

        # LOOP-SAFE HARD LIMITS (important)
        eff_basal = float(np.clip(eff_basal, 0.05, 2.5))
        eff_isf = float(np.clip(eff_isf, 10, 400))
        eff_cr = float(np.clip(eff_cr, 5, 200))

        eff_min_bg = float(eff_min_bg)
        eff_max_bg = float(eff_max_bg)

        if eff_min_bg >= eff_max_bg:
            eff_min_bg, eff_max_bg = 80.0, 180.0

        # -----------------------------
        # BUILD DOSE JSON
        # -----------------------------
        json_dose = get_json_loop_prediction_input_from_df(
            df_tail,
            eff_basal,
            eff_isf,
            eff_cr,
            prediction_start=datetime,
            insulin_type=self.insulin_type,
        )

        json_dose["target"] = [{
            "startDate": df_tail.index[0].strftime('%Y-%m-%dT%H:%M:%SZ'),
            "endDate":   df_tail.index[-1].strftime('%Y-%m-%dT%H:%M:%SZ'),
            "lowerBound": int(eff_min_bg),
            "upperBound": int(eff_max_bg),
        }]

        json_dose["maxBasalRate"] = float(np.clip(eff_basal * 2, 0.05, 10.0))
        json_dose["recommendationType"] = self.recommendation_type

        # FINAL CLEAN (prevents AST crash)
        #json_dose["maxBasalRate"] = sanitize(json_dose.get("maxBasalRate"))
        #json_dose["target"]["low"] = int(sanitize(json_dose["target"]["low"]))
        #json_dose["target"]["high"] = int(sanitize(json_dose["target"]["high"]))

        # -----------------------------
        # CRITICAL FIX: VALIDATE BEFORE CALL
        # -----------------------------
        if json_dose["maxBasalRate"] <= 0:
            self.add_patient_observation(name, datetime, glucose, basal_pump, 0.0, meal_grams, TDD)
            return Action(basal=basal_pump / 60, bolus=0.0)

        try:
            result = loop_to_python_api.get_dose_recommendations(json_dose)

            if not isinstance(result, dict):
                logger.warning("Invalid Loop output at BG=%.1f: %s", glucose, result)
                raise ValueError(f"Unexpected result type: {type(result)}")
                #return Action(basal=0.0, bolus=0.0)  # SAFE fallback: suspend, not deliver

        except Exception as e:
            logger.warning("Dose engine failed at BG=%.1f: %s", glucose, str(e))
            self.add_patient_observation(name, datetime, glucose, 0.0, 0.0, meal_grams, TDD)
            return Action(basal=0.0, bolus=0.0)  # SAFE fallback: suspend, not deliver

        # Also log basal_rec to see what the loop actually recommends:
        
        try:
            raw = result["automatic"]["basalAdjustment"]["unitsPerHour"]
            if raw is None:
                raise ValueError("unitsPerHour is None")
            basal_rec = float(raw)  # float(0) = 0.0, this is valid
        except Exception as e:
            basal_rec = 0.0 if glucose < 80 else basal_pump
            logger.warning("basal_rec parse failed at BG=%.1f (%s) — using %.4f", glucose, e, basal_rec)

        #logger.warning("LOOP OUTPUT: BG=%.1f basal_rec=%.4f result=%s", glucose, basal_rec, result)

        bolus_rec = 0.0
        if meal > 0:
            try:
                json_dose_meal = dict(json_dose)
                json_dose_meal["recommendationType"] = "manualBolus"
                meal_result = loop_to_python_api.get_dose_recommendations(json_dose_meal)
                bolus_rec = float(meal_result["manual"]["amount"])
            except Exception as e:
                logger.warning("Meal bolus failed at BG=%.1f: %s", glucose, e)
                bolus_rec = 0.0
        else:
            try:
                raw_auto_bolus = float(result["automatic"].get("bolusUnits", 0.0))

                # Cap automatic boluses during and after hypoglycemia recovery.
                # Post-suspend IOB underestimation causes the loop to recommend
                # large correction boluses when BG is still rising from a low.
                # This is clinically dangerous — real Loop app has a similar guard.
                if glucose < 100:
                    # BG still in recovery range — no automatic correction boluses
                    bolus_rec = 0.0
                    if raw_auto_bolus > 0:
                        logger.info(
                            "AUTO BOLUS SUPPRESSED during recovery: BG=%.1f raw=%.3fU",
                            glucose, raw_auto_bolus
                        )
                elif glucose < 130:
                    # BG rising but not fully stable — gentle cap
                    MAX_RECOVERY_BOLUS = 0.2
                    bolus_rec = min(raw_auto_bolus, MAX_RECOVERY_BOLUS)
                    if raw_auto_bolus > MAX_RECOVERY_BOLUS:
                        logger.info(
                            "AUTO BOLUS CAPPED near recovery: BG=%.1f raw=%.3fU → %.3fU",
                            glucose, raw_auto_bolus, bolus_rec
                        )
                else:
                    bolus_rec = raw_auto_bolus

            except Exception:
                bolus_rec = 0.0
        # This ensures IOB is computed from ACTUAL delivery, not scheduled basal
        self.add_patient_observation(
            name, datetime, glucose,
            basal=basal_rec,
            bolus=bolus_rec,
            carbs=meal_grams,
            TDD=TDD
        )

        # Record delivery into adaptive manager's df_history for sick detection.
        # This must happen AFTER basal_rec/bolus_rec are known.
        # _update_sick_flag reads basal+bolus from df_history to compute
        # total insulin delivery vs pump baseline.
        self.manager.record_delivery(name, datetime, basal_rec, bolus_rec)

        return Action(
            basal=basal_rec / 60.0,
            bolus=bolus_rec/ env_sample_time,
        )

    

    

    # unchanged API
    def get_current_effective_isf(self, patient_name: str) -> float:
        return self.manager.get_current_effective_isf(patient_name)

    def get_current_pump_isf(self, patient_name: str) -> float:
        return self.manager.get_current_pump_isf(patient_name)
    
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
    
    def get_current_autosens_ratio(self, patient_name: str) -> float:
        return self.manager.get_current_autosens_ratio(patient_name)

    def get_current_effective_isf(self, patient_name: str) -> float:
        return self.manager.get_current_effective_isf(patient_name)

    def get_current_pump_isf(self, patient_name: str) -> float:
        return self.manager.get_current_pump_isf(patient_name)

    def reset(self):
        super().reset()
        self.manager.reset()