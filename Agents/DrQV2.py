# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
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
        self.encoder.__call__ = lambda obs: self.aug(self.encoder(obs))
