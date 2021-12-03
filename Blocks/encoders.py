# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn
import Utils


class CNNEncoder(nn.Module):
    """CNN encoder."""

    def __init__(self, obs_shape, out_channels=32, depth=3, flatten=True, target_tau=None, optim_lr=None):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.out_channels = out_channels
        self.depth = depth
        self.flatten = flatten

        out_height, out_width = Utils.conv_output_shape(*obs_shape[-2:], kernel_size=3, stride=2)
        for _ in range(depth):
            out_height, out_width = Utils.conv_output_shape(out_height, out_width, kernel_size=3, stride=1)

        self.repr_shape = (out_channels,) + (out_height, out_width)
        self.repr_dim = out_channels * out_height * out_width

        self.conv_net = nn.Sequential(nn.Conv2d(obs_shape[0], out_channels, 3, stride=2), nn.ReLU(),
                                      *sum([(nn.Conv2d(out_channels, out_channels, 3, stride=1), nn.ReLU())
                                            for _ in range(depth)], ()))

        self.apply(Utils.weight_init)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(obs_shape=obs_shape, out_channels=out_channels, depth=depth, flatten=flatten)
            target.load_state_dict(self.state_dict())
            self.target = target

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, obs):
        # Operates on last 3 dims of obs, preserves leading dims
        shape = obs.shape
        obs = obs.reshape(-1, *self.obs_shape)
        # Assumes pixels
        obs = obs / 255.0 - 0.5
        h = self.conv_net(obs)
        if self.flatten:
            h = h.view(*shape[:-3], -1)
            assert h.shape[-1] == self.repr_dim
        else:
            h = h.view(*shape[:-3], *h.shape[-3:])
            assert tuple(h.shape[-3:]) == self.repr_shape
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
            target = self.__class__(in_dim=in_dim, feature_dim=feature_dim, hidden_dim=hidden_dim, out_dim=out_dim)
            target.load_state_dict(self.state_dict())
            self.target = target

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, *x):
        h = self.trunk(torch.cat(x, -1))

        return self.net(h)


class IsotropicCNNEncoder(nn.Module):
    """Isotropic (no bottleneck / dimensionality conserving) CNN encoder."""

    def __init__(self, obs_shape, action_dim=0, out_channels=64, flatten=True, target_tau=None, optim_lr=None):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.out_channels = out_channels
        self.flatten = flatten

        out_height, out_width = Utils.conv_output_shape(*obs_shape[-2:], kernel_size=(3, 3), pad=1)

        self.repr_shape = (out_channels,) + (out_height, out_width)
        self.repr_dim = out_channels * out_height * out_width

        assert obs_shape[-2] == out_height
        assert obs_shape[-1] == out_width

        self.conv_net = nn.Sequential(nn.Conv2d(obs_shape[0] + action_dim, out_channels, (3, 3), padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU(),
                                      nn.Conv2d(out_channels, out_channels, (3, 3), padding=1), nn.ReLU())

        self.apply(Utils.weight_init)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(obs_shape=obs_shape, action_dim=action_dim,
                                    out_channels=out_channels, flatten=flatten)
            target.load_state_dict(self.state_dict())
            self.target = target

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, obs, action=None):
        # Operates on last 3 dims of obs, preserves leading dims
        obs_shape = obs.shape
        obs = obs.reshape(-1, *self.obs_shape)

        if action is not None:
            # Appends action to channels
            action = action.view(-1, action.shape[-1])[:, :, None, None].expand(-1, -1, *obs.shape[-2:])
            obs = torch.cat([obs,  action], -3)

        h = self.conv_net(obs)

        if self.flatten:
            h = h.view(*obs_shape[:-3], -1)
            assert h.shape[-1] == self.repr_dim
        else:
            # h = h.view(*obs_shape[:-3], *h.shape[-3:])
            h = h.view(*obs_shape)  # Isotropic
            assert tuple(h.shape[-3:]) == self.repr_shape

        return h
