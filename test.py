import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
from ppo import IrrigationPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather
from pathlib import Path
from aquacropgymnasium.environment import Wheat

TRAIN_DIR = Path("train_output")
EVAL_DIR = './test_output'
WEATHER_PATH = Path("weather_data/champion_climate.txt")
os.makedirs(EVAL_DIR, exist_ok=True)

SEEDS = [1, 2, 3]
TIMESTEPS = [1_000_000, 2_000_000, 4_000_000]
EVAL_EPISODES = 100


def set_seed(seed):
    [seed_function(seed) for seed_function in [np.random.seed, random.seed, torch.manual_seed]]


def water_efficiency(yield_tonnes, irrigation_mm):
    tonnes_to_kg = 1000
    if irrigation_mm <= 0 or yield_tonnes < 0:
        return 0.00
    return (yield_tonnes * tonnes_to_kg ) / irrigation_mm


class random_action:
    def __init__(self, action_space):
        self.action_space = action_space
        self.predict = lambda obs, **kwargs: ([self.action_space.sample()], None)


class environment_configuration:
    def __init__(self, sim_start=2008, sim_end=2018):
        self.sim_start = sim_start
        self.sim_end = sim_end
    
    def create_env(self, seed):
        set_seed(seed)
        env = Monitor(Wheat(sim_start=self.sim_start, sim_end=self.sim_end))
        env.reset()
        return env
    
    def setup_vectorized_env(self, seed):
        return DummyVecEnv([lambda: self.create_env(seed)])


def run_single_episode(agent, env, max_steps=1000):
    obs = env.reset()
    done, reward_sum, steps = False, 0, 0
    
    while not done and steps < max_steps:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        reward_sum += reward[0]
        steps += 1
    
    return info[0], reward_sum

def load_model(seed, timestep):
    set_seed(seed)
    env_config = environment_configuration()
    env = env_config.setup_vectorized_env(seed)
    
    train_dir = Path(TRAIN_DIR)
    norm_file = train_dir / f"ppo_{timestep}_norm.pkl"
    model_file = train_dir / f"ppo_{timestep}"
    
    if norm_file.exists():
        env = VecNormalize.load(str(norm_file), env)
        env.training = False
        env.norm_reward = False
    
    model = IrrigationPPO.load(str(model_file), env=env)
    return model, env

def evaluate(agent, env, episodes=EVAL_EPISODES):
    results = []
    for _ in range(episodes):
        info, reward_sum = run_single_episode(agent, env)
        results.append({
            'yield': info.get('dry_yield', 0),
            'irrigation': info.get('total_irrigation', 0),
            'reward': reward_sum,
            'efficiency': water_efficiency(info.get('dry_yield', 0), info.get('total_irrigation', 0))
        })
    return pd.DataFrame(results)



def evaluate_mean_std(agent_type, df):
    return {
        'agent': agent_type,
        'mean_yield': df['yield'].mean(),
        'std_yield': df['yield'].std(),
        'mean_irrigation': df['irrigation'].mean(),
        'std_irrigation': df['irrigation'].std(),
        'water_efficiency': df['efficiency'].mean()
    }


def evaluate_heuristics():
    start_date = "2008/01/01"
    end_date = "2018/12/31"
    crop = Crop('Wheat', planting_date='01/01')
    weather_df = prepare_weather(WEATHER_PATH)
    soil = Soil('Loam')
    init_wc = InitialWaterContent(value=['FC'])

    strategies = {
        'Thresholds': IrrigationManagement(irrigation_method=1, SMT=[40, 60] * 8),
        'Interval': IrrigationManagement(irrigation_method=2, IrrInterval=7),
        'Rainfed': IrrigationManagement(irrigation_method=0)
    }

    results = []
    for name, strategy in strategies.items():
        model = AquaCropModel(
            start_date, end_date, weather_df, soil, crop,
            initial_water_content=init_wc, irrigation_management=strategy
        )
        model.run_model(till_termination=True)
        stats = model._outputs.final_stats
        results.append({
            'agent': name,
            'mean_yield': stats['Dry yield (tonne/ha)'].mean(),
            'std_yield': stats['Dry yield (tonne/ha)'].std(),
            'mean_irrigation': stats['Seasonal irrigation (mm)'].mean(),
            'std_irrigation': stats['Seasonal irrigation (mm)'].std(),
            'water_efficiency': water_efficiency(
                stats['Dry yield (tonne/ha)'].mean(),
                stats['Seasonal irrigation (mm)'].mean()
            )
        })
    return results



def generate_charts(df):
    blue = '#1f77b4'
    sns.set_style('white')
    plt.rcParams.update({'font.size': 14})

    plots = [
        ('mean_yield', 'std_yield', 'Mean Yield (tonne/ha)', 'combined_yields.png'),
        ('mean_irrigation', 'std_irrigation', 'Total Irrigation (mm)', 'combined_irrigations.png'),
        ('water_efficiency', None, 'Water Efficiency (kg/m³)', 'combined_water_efficiency.png')
    ]

    for metric, error_metric, ylabel, filename in plots:
        plt.figure(figsize=(12, 7))
        plot_colors = [blue] * len(df)

        if error_metric:
            bars = plt.bar(df['agent'], df[metric], yerr=df[error_metric], capsize=5, color=plot_colors, edgecolor='black')
        else:
            bars = plt.bar(df['agent'], df[metric], color=plot_colors, edgecolor='black')

        for bar, value in zip(bars, df[metric]):
            plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.05,
                     f'{value:.2f}', ha='center', va='bottom', fontsize=12)

        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, filename), dpi=300)
        plt.close()


def main():
    all_summary = []

    all_ppo_dfs = []
    per_timestep_summary = []

    for timestep in TIMESTEPS:
        for seed in SEEDS:
            try:
                model, env = load_model(seed, timestep)
                df = evaluate(model, env)
                df['timestep'] = timestep
                df['seed'] = seed
                all_ppo_dfs.append(df)
            except Exception as e:
                print(f"Error loading PPO model at timestep {timestep} seed {seed}: {e}")

    if all_ppo_dfs:
        combined_ppo_df = pd.concat(all_ppo_dfs)


        for timestep in TIMESTEPS:
            df = combined_ppo_df[combined_ppo_df['timestep'] == timestep]
            per_timestep_summary.append(evaluate_mean_std(f'PPO_{timestep // 1_000_000}M', df))


        ppo_timestep_df = pd.DataFrame(per_timestep_summary)
        ppo_timestep_df.to_csv(os.path.join(EVAL_DIR, "ppo_timesteps_averaged.csv"), index=False)

        combined_ppo_df.to_csv(os.path.join(EVAL_DIR, "ppo_all_detailed.csv"), index=False)

  
        best_timestep = max(per_timestep_summary, key=lambda x: x['mean_yield'])
        best_timestep['agent'] = 'PPO'  
        all_summary.append(best_timestep)


    rand_dfs = []
    for seed in SEEDS:
        env = setup_env(seed)
        agent = random_action(env.action_space)
        df = evaluate(agent, env)
        rand_dfs.append(df)
    rand_combined = pd.concat(rand_dfs)
    all_summary.append(evaluate_mean_std('Random', rand_combined))


    heuristics = evaluate_heuristics()
    all_summary.extend(heuristics)


    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(os.path.join(EVAL_DIR, "comparison_results.csv"), index=False)

    generate_charts(summary_df)

    print(summary_df[['agent', 'mean_yield', 'mean_irrigation', 'water_efficiency']].round(2))



if __name__ == "__main__":
    main()
