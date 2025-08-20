import warnings
import numpy as np
import torch as th
from collections import deque
from gymnasium import spaces
from stable_baselines3 import PPO 
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_schedule_fn


class IrrigationPPO(PPO):
    def __init__(self, *args, clip_range=0.2, ent_coef=0.0, **kwargs):
        if callable(clip_range):
            clip_range_val = clip_range(1.0)  
        else:
            clip_range_val = clip_range

        if callable(ent_coef):
            ent_coef_val = ent_coef(1.0)
        else:
            ent_coef_val = ent_coef

        self.clip_range_base = clip_range_val
        self.ent_coef_base = ent_coef_val

        super().__init__(*args, clip_range=clip_range, ent_coef=ent_coef, **kwargs)

        self.entropy_coef = ent_coef_val
        self.performance_history = deque(maxlen=50)
        self.kl_history = deque(maxlen=20)
        self.last_loss = np.nan

    def _clip_schedule(self, base_value):
        def clip(progress):
            return base_value
        return clip

    def _setup_model(self):
        super()._setup_model()
        self.clip_range = self._clip_schedule(self.clip_range_base)
        self.ent_coef = get_schedule_fn(self.ent_coef_base)

    def train(self):
        clip_range = self.clip_range(self._current_progress_remaining)
        ent_coef = self.ent_coef(self._current_progress_remaining)

        self.policy.set_training_mode(True)
        entropy_losses, pg_losses, value_losses, approx_kls = [], [], [], []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                pg_loss1 = advantages * ratio
                pg_loss2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                pg_loss = -th.min(pg_loss1, pg_loss2).mean()

                if self.clip_range_vf is None:
                    value_loss = th.nn.functional.mse_loss(rollout_data.returns, values)
                else:
                    clipped_values = rollout_data.old_values + (values - rollout_data.old_values).clamp(
                        -self.clip_range_vf, self.clip_range_vf)
                    value_loss = 0.5 * th.max(
                        (values - rollout_data.returns).pow(2),
                        (clipped_values - rollout_data.returns).pow(2)
                    ).mean()

                entropy_loss = -th.mean(entropy)
                loss = pg_loss + ent_coef * entropy_loss + self.vf_coef * value_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                entropy_losses.append(entropy_loss.item())
                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                approx_kl = th.mean(rollout_data.old_log_prob - log_prob).item()
                approx_kls.append(approx_kl)


        policy_loss = np.mean(pg_losses)
        entropy_loss = np.mean(entropy_losses)
        value_loss = np.mean(value_losses)


        self.last_loss = policy_loss + ent_coef * entropy_loss + self.vf_coef * value_loss
        self.logger.record("train/policy_gradient_loss", policy_loss)
        self.logger.record("train/value_loss", value_loss)
        self.logger.record("train/entropy_loss", entropy_loss)
        self.logger.record("train/approx_kl", np.mean(approx_kls))
