# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import Utils
from Agents.DQNDPG import DQNDPGAgent


class DrQV2Agent(DQNDPGAgent):
    """Data-Regularized Q-Network V2"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 device, log_tensorboard  # On-boarding
                 ):
        # discrete = False
        super().__init__(
            obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
            lr, target_tau,  # Optimization
            explore_steps, stddev_schedule, stddev_clip,  # Exploration
            False, device, log_tensorboard  # On-boarding
        )
        # Explores via a defined schedule
        assert Utils.schedule(stddev_schedule, 0)
