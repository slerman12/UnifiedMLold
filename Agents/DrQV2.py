# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import Utils

from Agents import DQNDPGAgent
from Blocks.actors import TruncatedGaussianActor

from Blocks.augmentations import RandomShiftsAug, IntensityAug


class DrQV2Agent(DQNDPGAgent):
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, update_per_steps, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, device, log_tensorboard  # On-boarding
                 ):
        super().__init__(
            obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
            lr, update_per_steps, target_tau,  # Optimization
            explore_steps, stddev_schedule, stddev_clip,  # Exploration
            discrete, device, log_tensorboard  # On-boarding
        )

        # ! Technically DrQV2 only compatible with continuous spaces but both supported here
        # self.discrete = False  # Discrete supported
        # self.actor = TruncatedGaussianActor(self.encoder.repr_dim, feature_dim, hidden_dim, action_shape[-1],
        #                                     stddev_schedule, stddev_clip,
        #                                     optim_lr=lr).to(device)

        # Data augmentation
        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

    def batch_processing(self, *batch, logs=None):
        obs, action, reward, discount, next_obs, *traj = Utils.to_torch(batch, self.device)

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        return super().batch_processing(obs, action, reward, discount, next_obs, *traj,
                                        logs=logs)
