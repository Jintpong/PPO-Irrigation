import os
import torch
import tensorboard
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import List
from dataclasses import dataclass
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
from ppo import IrrigationPPO
from aquacropgymnasium.environment import Wheat

@dataclass
class Config:
    timesteps = None
    lr = 2.5e-4
    n_steps = 4096
    vf_coef = 0.5
    batch_size = 512
    n_epochs = 23
    gamma = 0.98
    clip_range: callable = lambda progress: 0.2
    ent_coef = 0.01

    def __post_init__(self):
        if self.timesteps is None:
            self.timesteps = [1_000_000, 2_000_000, 4_000_000]


class RewardTracker(BaseCallback):
    def __init__(self, name, save_dir):
        super().__init__()
        self.name = name
        self.save_dir = Path(save_dir)
        self.ep_rewards = []
        self.ep_timesteps = []
        self.writer = SummaryWriter(log_dir=self.save_dir / f"runs/{self.name}")

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_reward = info['episode']['r']
                self.ep_rewards.append(ep_reward)
                self.ep_timesteps.append(self.num_timesteps)
                self.writer.add_scalar("Episode/MeanReward", ep_reward, self.num_timesteps)
        return True

    def _on_training_end(self):
        if not self.ep_rewards:
            return

        rewards = np.array(self.ep_rewards)
        timesteps = np.array(self.ep_timesteps)
        print(f"{self.name} - Mean: {rewards.mean():.2f}, Std: {rewards.std():.2f}")

        if len(rewards) >= 10:
            smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
            smoothed_timesteps = timesteps[len(timesteps) - len(smoothed_rewards):]
        else:
            smoothed_rewards = rewards
            smoothed_timesteps = timesteps

        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_timesteps, smoothed_rewards, label=f"{self.name}", color='tab:blue')
        plt.xlabel("Timesteps")
        plt.ylabel("Average Reward")
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        plt.grid(True)
        plt.legend()

        np.save(self.save_dir / f"{self.name}_rewards.npy", rewards)
        np.save(self.save_dir / f"{self.name}_timesteps.npy", timesteps)
        plt.savefig(self.save_dir / f"{self.name}_rewards.png", dpi=300)
        plt.close()

        self.writer.close()


def make_env():
    return Monitor(Wheat(sim_start=1982, sim_end=2007))

def train_ppo(timesteps, config, output_dir):
    name = f"ppo_{timesteps}"
    print(f"\nTraining {name} for {timesteps:,} steps...")

    env = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=True)
    tracker = RewardTracker(name, output_dir)

    model = IrrigationPPO("MlpPolicy", env, learning_rate=config.lr, n_steps=config.n_steps,
                vf_coef=config.vf_coef, batch_size=config.batch_size,
                n_epochs=config.n_epochs, gamma=config.gamma,
                clip_range=config.clip_range, ent_coef=config.ent_coef, verbose=1)

   
    loss_values = []
    loss_timesteps = []

    total_timesteps = 0
    callback = tracker

 
    while total_timesteps < timesteps:
        update_timesteps = min(config.n_steps, timesteps - total_timesteps)
        model.learn(total_timesteps=update_timesteps, callback=callback, reset_num_timesteps=False)
        total_timesteps += update_timesteps


        if hasattr(model, "last_loss"):
            loss = model.last_loss  
        else:
            loss = np.nan 

        loss_values.append(loss)
        loss_timesteps.append(total_timesteps)

    model._callbacks = None
    model.save(output_dir / f"{name}.zip")

    try:
        env.save(output_dir / f"{name}_norm.pkl")
        print("Model and normalization saved")
    except Exception as e:
        print(f"Save error: {e}")

    
    np.save(output_dir / f"{name}_loss.npy", np.array(loss_values))
    np.save(output_dir / f"{name}_loss_timesteps.npy", np.array(loss_timesteps))

    env.close()



def plot_combined_rewards(timesteps_list, output_dir):
    for timesteps in timesteps_list:
        name = f"ppo_{timesteps}"
        reward_path = output_dir / f"{name}_rewards.npy"
        timestep_path = output_dir / f"{name}_timesteps.npy"
        loss_path = output_dir / f"{name}_loss.npy"
        loss_timestep_path = output_dir / f"{name}_loss_timesteps.npy"

        if not reward_path.exists() or not timestep_path.exists():
            print(f"Warning: reward/timestep data missing for {name}")
            continue

        rewards = np.load(reward_path)
        timesteps_data = np.load(timestep_path)

      
        if len(rewards) >= 10:
            smoothed_rewards = np.convolve(rewards, np.ones(10) / 10, mode='valid')
            smoothed_timesteps = timesteps_data[-len(smoothed_rewards):]
        else:
            smoothed_rewards = rewards
            smoothed_timesteps = timesteps_data

        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        
        ax1.plot(smoothed_timesteps, smoothed_rewards, label="Average Reward", color='tab:blue')
        ax1.set_xlabel("Timesteps", fontsize=12, color='black')
        ax1.set_ylabel("Average Reward", fontsize=12, color='black')
        ax1.tick_params(axis='x', colors='black')
        ax1.tick_params(axis='y', colors='black')
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

      
        if loss_path.exists() and loss_timestep_path.exists():
            losses = np.load(loss_path)
            loss_timesteps = np.load(loss_timestep_path)

            if len(losses) >= 10:
                smoothed_losses = np.convolve(losses, np.ones(10) / 10, mode='valid')
                smoothed_loss_timesteps = loss_timesteps[-len(smoothed_losses):]
            else:
                smoothed_losses = losses
                smoothed_loss_timesteps = loss_timesteps

            ax2.plot(smoothed_loss_timesteps, smoothed_losses, linestyle='--', color='tab:red', label="Training Loss")
            ax2.set_ylabel("Training Loss", fontsize=12, color='black')
            ax2.tick_params(axis='y', colors='black')


        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines + lines2, labels + labels2)
        for text in legend.get_texts():
            text.set_color("black")


        for spine in ax1.spines.values():
            spine.set_edgecolor('black')
        for spine in ax2.spines.values():
            spine.set_edgecolor('black')

        fig.tight_layout()
        ax1.grid(True)

        plt.savefig(output_dir / f"{name}_training_results.png", dpi=300)
        plt.close()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = Config()
    output_dir = Path('./train_output')
    output_dir.mkdir(exist_ok=True)

    for timesteps in config.timesteps:
        train_ppo(timesteps, config, output_dir)

    plot_combined_rewards(config.timesteps, output_dir)
    print("\nAll training completed!")


if __name__ == "__main__":
    main()
