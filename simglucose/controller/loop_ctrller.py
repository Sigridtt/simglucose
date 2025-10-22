from .base import Controller
from .base import Action
from loop_to_python_api.helpers import get_json_loop_prediction_input_from_df
import loop_to_python_api.api as loop_to_python_api
import numpy as np
import pandas as pd
import pkg_resources
import logging

logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename('simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename('simglucose', 'params/vpatient_params.csv')


class LoopController(Controller):
    """
    This is the LoopAlgorithm set with some basic settings.
    """

    def __init__(self, target=140, recommendation_type='tempBasal', use_tdd_settings=False,
                 use_fully_closed_loop=False, insulin_type='novolog'):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.target = target
        self.observations = {}
        self.recommendation_type = recommendation_type
        self.use_tdd_settings = use_tdd_settings
        self.use_fully_closed_loop = use_fully_closed_loop
        self.insulin_type = insulin_type

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')  # unit: g/min
        datetime = kwargs.get('time')

        action = self._loop_policy(datetime, pname, meal, observation.CGM, sample_time)
        return action

    def _loop_policy(self, datetime, name, meal, glucose, env_sample_time, TDD=None):
        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(
                name)]
            u2ss = params.u2ss.values.item()  # unit: pmol/(L*kg)
            BW = params.BW.values.item()  # unit: kg
            TDD = quest.TDI.values[0]
        else:
            quest = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
                                 columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43  # unit: pmol/(L*kg)
            BW = 57.0  # unit: kg
            if TDD is None:
                TDD = 50

        if self.use_tdd_settings:
            basal_pr_hr, isf, cr = self.get_therapy_settings_from_tdd(TDD)
            basal = basal_pr_hr / 60  # unit: U/hr
        else:
            basal = u2ss * BW / 6000  # unit: U/min
            basal_pr_hr = basal * 60  # unit: U/hr
            cr = float(quest.CR.values[0])
            isf = float(quest.CF.values[0])

        meal_grams = meal * env_sample_time  # From g/min to g

        if self.use_fully_closed_loop:
            meal_grams = 0
            meal = 0

        # Load previous observations for patient and add the new CGM observation
        # TODO: Shouldn't observations be fetched after we add the new glucose and meal reading?
        df_observations = self.get_patient_observations(key=name)
        self.add_patient_observation(name, datetime, glucose, np.nan, np.nan, meal_grams)

        # If observations for < 3 hrs, return basal=scheduled basal and bolus=0
        if len(df_observations) < (3 * 60 // env_sample_time):
            self.add_patient_observation(name, datetime, glucose, basal_pr_hr, 0, meal_grams)
            return Action(basal=basal, bolus=0)

        # Get data input for the Loop Algorithm insulin recommendation
        df_observations = df_observations.sort_index().tail(int(12*60 // env_sample_time))
        json_data = get_json_loop_prediction_input_from_df(df_observations, basal_pr_hr, isf, cr,
                                                           prediction_start=datetime, insulin_type=self.insulin_type)

        # Setting max basal to the double of the scheduled basal rate
        json_data['maxBasalRate'] = basal_pr_hr * 2
        json_data['recommendationType'] = self.recommendation_type  # Can be: "automaticBolus", "tempBasal"
        dose_recommendations = loop_to_python_api.get_dose_recommendations(json_data)
        basal_rec = dose_recommendations['automatic']['basalAdjustment']['unitsPerHour']

        if meal > 0:
            # Add manual bolus for meals. Algorithm does not recommend meal boluses if we do not do this
            json_data['recommendationType'] = 'manualBolus'
            dose_recommendations = loop_to_python_api.get_dose_recommendations(json_data)
            bolus_rec = dose_recommendations['manual']['amount']
        elif 'bolusUnits' in dose_recommendations['automatic']:
            bolus_rec = dose_recommendations['automatic']['bolusUnits']
        else:
            bolus_rec = 0.0

        # Overwrite patient data iteration with insulin action
        self.add_patient_observation(name, datetime, glucose, basal=basal_rec, bolus=bolus_rec, carbs=meal_grams)

        # This is to convert basal (U/hr) and bolus (U) to insulin rate (U/min), as required by the simulation env
        action = Action(basal=basal_rec / 60, bolus=bolus_rec / env_sample_time)
        return action

    def reset(self):
        pass

    def get_patient_observations(self, key: str) -> pd.DataFrame:
        if key in self.observations:
            return self.observations[key]
        return pd.DataFrame(columns=["CGM", "basal", "bolus", "carbs"], index=pd.DatetimeIndex([], name="date"))

    def add_patient_observation(self, key: str, datetime, cgm, basal, bolus, carbs):
        """Add a row to the state dataframe under the given key."""
        new_data = {
            "CGM": cgm,
            "basal": basal,
            "bolus": bolus,
            "carbs": np.nan if carbs <= 0 else carbs
        }
        df = self.get_patient_observations(key)
        df.loc[datetime] = new_data
        self.observations[key] = df

