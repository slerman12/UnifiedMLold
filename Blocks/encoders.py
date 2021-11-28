# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn
import Utils


class CNNEncoder(nn.Module):
    def __init__(self, obs_shape, target_tau=None, optim_lr=None):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape

        self.repr_dim = 32 * 35 * 35
        self.repr_shape = (32, 35, 35)

        self.conv_net = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                      nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                      nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                      nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                      nn.ReLU())

        self.apply(Utils.weight_init)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(obs_shape)
            target.load_state_dict(self.state_dict())
            self.target = target

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, obs, flatten=True):
        # operates on last 3 dims of obs, preserves leading dims
        shape = obs.shape
        obs = obs.view(-1, *self.obs_shape)
        obs = obs / 255.0 - 0.5
        h = self.conv_net(obs)
        if flatten:
            h = h.view(*shape[:-3], -1)
        else:
            h = h.view(*shape[:-3], *h.shape[-3:])
        return h


class LayerNormMLPEncoder(nn.Module):
    """Layer-norm MLP encoder."""
    def __init__(self, in_dim, feature_dim, hidden_dim, out_dim,
                 target_tau=None, optim_lr=None):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(in_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim))

        self.apply(Utils.weight_init)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(in_dim, feature_dim, hidden_dim, out_dim)
            target.load_state_dict(self.state_dict())
            self.target = target

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, *x):
        h = self.trunk(torch.cat(x, -1))

        return self.net(h)

