# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

import Utils

from Agents.DQNDPG import DQNDPGAgent

from Blocks.augmentations import RandomShiftsAug, IntensityAug

from Losses.PolicyLearning import deepPolicyGradient
from Losses.QLearning import ensembleQLearning


class DrQV2Agent(DQNDPGAgent):
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, device, log_tensorboard  # On-boarding
                 ):
        super().__init__(
            obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
            lr, target_tau,  # Optimization
            stddev_schedule, stddev_clip,  # Exploration
            discrete, device, log_tensorboard  # On-boarding
        )
        self.explore_steps = explore_steps  # 2000

        # ! Technically DrQV2 only compatible with continuous spaces but both supported here
        # self.discrete = False  # Discrete supported

        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

    def act(self, obs):
        action = super().act(obs)

        # Explore phase
        if self.step < self.explore_steps and self.training:
            action = torch.randint(self.actor.action_dim, size=action.shape) if self.discrete \
                else action.uniform_(-1, 1)

        return action

    # Data augmentation
    def see_augmented(self, obs):
        obs = self.aug(obs)
        obs = self.encoder(obs)
        return obs

    def update(self, replay):
        logs = {'episode': self.episode, 'step': self.step} if self.log_tensorboard \
            else None

        batch = replay.sample()  # Can also write 'batch = next(replay)'
        obs, action, reward, discount, next_obs, *traj = Utils.to_torch(
            batch, self.device)

        # "See" augmented
        obs = self.see_augmented(obs)
        with torch.no_grad():
            next_obs = self.see_augmented(next_obs)

        if self.log_tensorboard:
            logs['batch_reward'] = reward.mean().item()

        # Critic loss
        critic_loss = ensembleQLearning(self.actor, self.critic,
                                        obs, action, reward, discount, next_obs,
                                        self.step, logs=logs)

        # Update critic
        Utils.optimize(critic_loss,
                       self.encoder,
                       self.critic)

        self.critic.update_target_params()

        # Actor loss
        actor_loss = deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                        self.step, logs=logs)

        # Update actor
        Utils.optimize(actor_loss,
                       self.actor)

        return logs
