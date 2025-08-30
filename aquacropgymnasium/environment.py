import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from aquacrop.core import AquaCropModel
from aquacrop.entities.crop import Crop
from aquacrop.entities.soil import Soil
from aquacrop.entities.irrigationManagement import IrrigationManagement
from aquacrop.entities.inititalWaterContent import InitialWaterContent
from pathlib import Path
from aquacrop.utils import prepare_weather

class Wheat(gym.Env):
    def __init__(self, sim_start=1982, sim_end=2018, crop='Wheat', climate_file='champion_climate.txt', planting_date='01/01'):
        super().__init__()
        
        self.year_start, self.year_end = sim_start, sim_end
        self.crop_name, self.planting_schedule = crop, planting_date
        self.weather_file = climate_file
        

        current_dir = os.path.dirname(os.path.dirname(__file__))
        self.climate_file_path = f"{current_dir}{os.sep}weather_data{os.sep}{climate_file}"
        self.soil_config = Soil('Loam')
        self.water_init = InitialWaterContent(value=['FC'])
        

        self.min_irrigation, self.max_irrigation, self.irrigation_steps = 0, 25, 2
        step_size = (self.max_irrigation - self.min_irrigation) / (self.irrigation_steps - 1)
        self.action_depths = [self.min_irrigation + i * step_size for i in range(self.irrigation_steps)]
        
        self.action_space = spaces.Discrete(self.irrigation_steps)
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(26,), dtype=np.float32)

    def get_current_observation(self):
        condition = self.model._init_cond
        age = condition.age_days 
        canopy = condition.canopy_cover  
        biomass_log = condition.biomass
        taw_value = condition.taw if condition.taw > 0 else 1.0
        depletion_ratio = condition.depletion / taw_value
        taw_raw = condition.taw
        crop_features = [age, canopy, biomass_log, depletion_ratio, taw_raw]
        

        weather_features = []
        current_step = self.model._clock_struct.time_step_counter
        for var_name in ['Precipitation', 'MinTemp', 'MaxTemp']:
            window_start = max(0, current_step - 7)
            window_end = current_step
            
            if window_end > window_start:
                data = self.weather_df[var_name].iloc[window_start:window_end].values
            else:
                data = np.array([])
            

            if len(data) < 7:
                fill_value = np.median(data) if len(data) > 0 else 0.0
                padded = [fill_value] * (7 - len(data)) + data.tolist()
            else:
                padded = data.tolist()
            
            normalized = padded
            
            weather_features.extend(normalized)
        
        all_features = crop_features + weather_features
        observation = np.array(all_features, dtype=np.float32)
        
        if len(observation) < 26:
            padding = self.rng.normal(0, 0.01, 26 - len(observation)).astype(np.float32)
            observation = np.concatenate([observation, padding])
        elif len(observation) > 26:
            observation = observation[:26]
            
        return observation

    def initialize_environment(self, seed=None, options=None):
        super().reset(seed=seed)
    
        self.rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        self.simcalyear = self.rng.randint(self.year_start, self.year_end + 1)
        self.crop = Crop(self.crop_name, self.planting_schedule)
        self.irrigation_accumulator = self.reward_accumulator = self.step_counter = 0.0
        
        p = Path(self.climate_file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Cannot locate weather file: {p}")
        self.weather_df = prepare_weather(p)
        
        start_date = f'{self.simcalyear}/{self.planting_schedule}'
        end_date = f'{self.simcalyear}/12/31'
        self.model = AquaCropModel(
            start_date, end_date, self.weather_df, self.soil_config, self.crop,
            irrigation_management=IrrigationManagement(irrigation_method=5),
            initial_water_content=self.water_init
        )
        self.model.run_model()
        
        return self.get_current_observation(), self._create_info_dict()

    def reset(self, *, seed=None, options=None):
        return self.initialize_environment(seed=seed, options=options)


    def step(self, action):
        action = max(0, min(action, len(self.action_depths) - 1))
        irrigation_amount = self.action_depths[action]
        self.model._param_struct.IrrMngt.depth = max(0.0, float(irrigation_amount))
        self.model.run_model(initialize_model=False)
        
        self.irrigation_accumulator += irrigation_amount
        self.step_counter += 1
        

        if irrigation_amount > 0:
            step_reward = -self.irrigation_accumulator
        else:
            step_reward = 0
        self.reward_accumulator += step_reward
        
        model_finished = self.model._clock_struct.model_is_finished
        truncated = False 
        
        if model_finished:
            final_results = self.model._outputs.final_stats
            yield_value = final_results['Dry yield (tonne/ha)'].mean()
            irrigation_total = final_results['Seasonal irrigation (mm)'].mean()
            
            yield_component = yield_value ** 4
            final_reward = self.reward_accumulator + yield_component 
            
            episode_info = {
                'dry_yield': float(yield_value),
                'total_irrigation': float(irrigation_total),
                'episode': {
                    'r': float(final_reward),
                    'l': int(self.step_counter),
                    'yield_reward': float(yield_component),
                    'irrigation_penalty': float(self.reward_accumulator)
                }
            }
            
            self.reward_accumulator = 0.0
            return self.get_current_observation(), final_reward, True, False, episode_info
        else:
            return self.get_current_observation(), step_reward, False, truncated, self._create_info_dict()

    def _create_info_dict(self):
        return {
            'dry_yield': 0.0,
            'total_irrigation': self.irrigation_accumulator,
            'steps_taken': int(self.step_counter),
            'current_year': self.simcalyear
        }

    def close(self):
        for attr in ['model', 'weather_df', 'rng']:
            if hasattr(self, attr):
                delattr(self, attr)
        self.irrigation_accumulator = self.reward_accumulator = self.step_counter = 0.0