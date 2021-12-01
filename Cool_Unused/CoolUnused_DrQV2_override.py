# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

from Cool_Unused.CoolUnused_DQNDPG_short import DQNDPGAgent

from Blocks.augmentations import RandomShiftsAug, IntensityAug


class DrQV2Agent(DQNDPGAgent):
    """Data-Regularized Q-Network V2"""
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
        # Optionally, trainable variance for policy supported here as well if stddev_schedule=null

        aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

        encode = self.encoder.forward

        # "See"
        def see_augmented(obs):
            if self.encoder.training:
                obs = aug(obs)
            obs = encode(obs)
            return obs

        # Data augmentation
        self.encoder.forward = see_augmented

    def act(self, obs):
        action = super().act(obs)

        # Explore phase
        if self.step < self.explore_steps and self.training:
            action = torch.randint(self.actor.action_dim, size=action.shape) if self.discrete \
                else action.uniform_(-1, 1)

        return action
