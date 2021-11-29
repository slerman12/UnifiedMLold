# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

import Utils

from Agents import DQNDPGAgent

from Blocks.augmentations import RandomShiftsAug, IntensityAug


class DrQV2Agent(DQNDPGAgent):
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 target_tau, stddev_schedule, stddev_clip,  # Models
                 lr, update_per_steps,  # Optimization
                 explore_steps,  # Exploration
                 discrete, device, log_tensorboard  # On-boarding
                 ):
        super().__init__(
                         obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                         target_tau, stddev_schedule, stddev_clip,  # Models
                         lr, update_per_steps,  # Optimization
                         explore_steps,  # Exploration
                         discrete, device, log_tensorboard  # On-boarding
                         )

        # ! Technically DrQV2 only compatible with continuous spaces but both supported here
        # self.discrete = False

        # Data augmentation
        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)
        # self.encoder.__call__ = lambda obs: self.aug(self.encoder(obs))

    def update(self, replay):
        logs = {'episode': self.episode, 'step': self.step}

        batch = replay.sample()
        obs, action, reward, discount, next_obs, *traj = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r = traj

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # Encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.log_tensorboard:
            logs['batch_reward'] = reward.mean().item()

        # Update critic
        self.update_critic(obs, action, reward, discount, next_obs,
                           logs=logs if self.log_tensorboard else None)

        # Update actor
        self.update_actor(obs,
                          logs=logs if self.log_tensorboard else None)

        # Update critic target
        self.critic.update_target_params()

        return logs
