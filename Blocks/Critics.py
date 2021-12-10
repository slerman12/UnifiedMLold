# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

import Utils

from Blocks.Architectures.MLP import MLP
from Blocks.Architectures.Residual import ResidualBlock


class _EnsembleQCritic(nn.Module):
    """Base critic"""
    def __init__(self):
        super().__init__()
        self.trunk = nn.Identity()
        self.Q_head = None

    def __post__(self, action_dim, ensemble_size, discrete, optim_lr=None, target_tau=None, **kwargs):

        assert self.Q_head is not None, 'Inheritor of Critic must define self.Q_head'

        # Initialize weights
        self.apply(Utils.weight_init)

        # Optimizer
        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        # EMA 
        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(action_dim=action_dim, ensemble_size=ensemble_size,
                                    discrete=discrete, **kwargs)
            target.load_state_dict(self.state_dict())
            self.target = target

        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.discrete = discrete

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    # Get action Q-values
    def forward(self, obs=None, action=None, dist=None):
        if self.discrete:
            assert obs is not None or dist is not None, 'Q-values require obs or existing dist'

            # All actions' Q-values
            if dist is None:
                h = self.trunk(obs)
                Qs = tuple(Q_net(h) for Q_net in self.Q_head)
            else:
                Qs = dist.Qs

            # Q-values for a discrete action
            if action is not None:
                ind = action.long().view(*Qs[0].shape[:-1], 1)
                Qs = tuple(torch.gather(Q, -1, ind) for Q in Qs)
        else:
            assert obs is not None and action is not None, 'Action must be specified for continuous'

            # Q-values for a continuous action
            h = self.trunk(obs)
            h_a = torch.cat([h, action], dim=-1)
            Qs = tuple(Q_net(h_a) for Q_net in self.Q_head)

        # Ensemble of Q-values
        return Qs


class MLPEnsembleQCritic(_EnsembleQCritic):
    """
    MLP-based Critic network, employs ensemble Q learning,
    e.g. DrQV2 (https://arxiv.org/abs/2107.09645).
    """

    def __init__(self, repr_shape, feature_dim, hidden_dim, action_dim, ensemble_size=2, critic_norm=False,
                 discrete=False, target_tau=None, optim_lr=None):

        super().__init__()

        repr_dim = math.prod(repr_shape)

        # Linear + LayerNorm
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        # MLP dimensions
        in_dim = feature_dim if discrete else feature_dim + action_dim
        Q_dim = action_dim if discrete else 1

        # MLP
        self.Q_head = nn.ModuleList([MLP(in_dim=in_dim,
                                         hidden_dim=hidden_dim,
                                         out_dim=Q_dim,
                                         depth=1,
                                         l2_norm=critic_norm)
                                     for _ in range(ensemble_size)])

        self.__post__(optim_lr=optim_lr, target_tau=target_tau, repr_shape=repr_shape,
                      feature_dim=feature_dim, hidden_dim=hidden_dim, action_dim=action_dim,
                      ensemble_size=ensemble_size, critic_norm=critic_norm, discrete=discrete)


class CNNEnsembleQCritic(_EnsembleQCritic):
    """
    CNN-based Critic network, employs ensemble Q learning,
    e.g. Efficient-Zero (https://arxiv.org/pdf/2111.00210.pdf) (except with ensembling).
    """

    def __init__(self, repr_shape, hidden_channels, out_channels, num_blocks,
                 hidden_dim, action_dim, ensemble_size=2, critic_norm=False,
                 discrete=False, target_tau=None, optim_lr=None):

        super().__init__()

        in_channels, height, width = repr_shape

        # CNN
        self.trunk = nn.Sequential(*[ResidualBlock(in_channels, hidden_channels)
                                     for _ in range(num_blocks)],
                                   nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Flatten())

        # CNN dimensions
        trunk_h, trunk_w = Utils.cnn_output_shape(height, width, self.trunk)
        feature_dim = out_channels * trunk_h * trunk_w

        # MLP dimensions
        in_dim = feature_dim if discrete else feature_dim + action_dim
        Q_dim = action_dim if discrete else 1

        # MLP
        self.Q_head = nn.ModuleList([MLP(in_dim, Q_dim, hidden_dim, 1)
                                     for _ in range(ensemble_size)])

        self.__post__(optim_lr=optim_lr, target_tau=target_tau, repr_shape=repr_shape,
                      hidden_channels=hidden_channels, out_channels=out_channels, num_blocks=num_blocks,
                      hidden_dim=hidden_dim, action_dim=action_dim, ensemble_size=ensemble_size,
                      critic_norm=critic_norm, discrete=discrete)

