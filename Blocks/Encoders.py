# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn

import Utils

from Blocks.Architectures.Residual import ResidualBlock, Residual


class BaseEncoder(nn.Module):
    """Base CNN encoder."""

    def __init__(self, obs_shape, context_dim=0, out_channels=32, depth=3, pixels=True, flatten=True,
                 target_tau=None, optim_lr=None, **kwargs):
        super().__init__()

        assert len(obs_shape) == 3

        self.obs_shape = obs_shape
        self.in_channels = in_channels = obs_shape[0] + context_dim
        self.out_channels = out_channels
        self.depth = depth

        self.pixels = pixels

        self.flatten = flatten

        self.kwargs = kwargs
        self.target_tau = target_tau
        self.optim_lr = optim_lr

        self.CNN = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=2), nn.ReLU(),
                                 *sum([(nn.Conv2d(out_channels, out_channels, 3, stride=1), nn.ReLU())
                                       for _ in range(depth)], ()))

    def init(self):
        self.apply(Utils.weight_init)

        # Pre-compute CNN feature map dimensions
        height, width = Utils.cnn_output_shape(self.obs_shape[1],
                                               self.obs_shape[2],
                                               self.CNN)
        self.repr_shape = (self.out_channels, height, width)
        self.repr_dim = self.out_channels * height * width

        # Set up optimizer
        if self.optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=self.optim_lr)

        # Set up EMA target
        if self.target_tau is not None:
            target = self.__class__(obs_shape=self.obs_shape, out_channels=self.out_channels,
                                    depth=self.depth, flatten=self.flatten, **self.kwargs)
            target.load_state_dict(self.state_dict())
            self.target = target

    # EMA self-copy
    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, obs, context=None):
        # Operates on last 3 dims of obs, preserves leading dims
        shape = obs.shape
        obs = obs.reshape(-1, *self.obs_shape)

        if self.pixels:
            # Normalizes pixels
            obs = obs / 255.0 - 0.5

        # Optionally append context to channels if dimensions allow
        if context is not None:
            assert context.shape[-1] == self.in_channels - obs.shape[1]
            # Appends context to channels
            action = context.reshape(-1, context.shape[-1])[:, :, None, None].expand(-1, -1, *obs.shape[-2:])
            obs = torch.cat([obs,  action], 1)

        # CNN encode
        encoding = self.CNN(obs)

        # Optionally flatten
        if self.flatten:
            encoding = encoding.view(*shape[:-3], -1)
            assert encoding.shape[-1] == self.repr_dim
        else:
            encoding = encoding.view(*shape[:-3], *encoding.shape[-3:])
            assert tuple(encoding.shape[-3:]) == self.repr_shape
        return encoding


class BasicCNNEncoder(BaseEncoder):
    """Basic CNN encoder, e.g., DrQV2 (https://arxiv.org/abs/2107.09645)."""

    def __init__(self, obs_shape, out_channels=32, depth=3, pixels=True, flatten=True,
                 target_tau=None, optim_lr=None):
        super().__init__(obs_shape=obs_shape, out_channels=out_channels, depth=depth,
                         pixels=pixels, flatten=flatten,
                         target_tau=target_tau, optim_lr=optim_lr)

        self.init()


class ResidualBlockEncoder(BaseEncoder):
    """Residual block-based CNN encoder, e.g., Efficient-Zero (https://arxiv.org/pdf/2111.00210.pdf)."""

    def __init__(self, obs_shape, context_dim=0, out_channels=64, pixels=True,  flatten=True, num_blocks=1,
                 target_tau=None, optim_lr=None, **kwargs):
        super().__init__(obs_shape=obs_shape, context_dim=context_dim, out_channels=out_channels,
                         depth=0, num_blocks=num_blocks,
                         pixels=pixels, flatten=flatten,
                         target_tau=target_tau, optim_lr=optim_lr)

        huge = False
        if huge:
            conv = nn.Conv2d(out_channels // 2, out_channels,
                             kernel_size=3, stride=2,
                             padding=1, bias=False)

            self.CNN = nn.Sequential(nn.Conv2d(self.in_channels, out_channels // 2,
                                               kernel_size=3, stride=2,
                                               padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels // 2),
                                     nn.ReLU(),
                                     ResidualBlock(out_channels // 2, out_channels // 2),
                                     ResidualBlock(out_channels // 2, out_channels,
                                                   stride=2, down_sample=conv),
                                     ResidualBlock(out_channels, out_channels),
                                     nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                                     ResidualBlock(out_channels, out_channels),
                                     nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                                     *[ResidualBlock(out_channels, out_channels)
                                       for _ in range(num_blocks)])
        else:
            self.CNN = nn.Sequential(nn.Conv2d(self.in_channels, out_channels,
                                               kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels // 2),
                                     nn.ReLU(),
                                     *[ResidualBlock(out_channels, out_channels)
                                       for _ in range(num_blocks)])

        self.init()


"""Creators: As in, "to create." Generative models that plan, forecast, and imagine."""


class IsotropicCNNEncoder(BaseEncoder):
    """Isotropic (no bottleneck / dimensionality conserving) CNN encoder,
    e.g., SPR (https://arxiv.org/pdf/2007.05929.pdf)."""

    def __init__(self, obs_shape, context_dim=0, out_channels=64, depth=0, pixels=False, flatten=True,
                 target_tau=None, optim_lr=None, **kwargs):
        super().__init__(obs_shape=obs_shape, context_dim=context_dim, out_channels=out_channels, depth=depth,
                         pixels=pixels, flatten=flatten,
                         target_tau=target_tau, optim_lr=optim_lr)

        self.CNN = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, (3, 3), padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 *sum([(nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU())
                                       for _ in range(depth)], ()),
                                 nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
                                 nn.ReLU())

        self.init()

        # Isotropic
        assert obs_shape[-2] == self.repr_shape[1]
        assert obs_shape[-1] == self.repr_shape[2]


class IsotropicResidualBlockEncoder(BaseEncoder):
    """Isotropic (no bottleneck / dimensionality conserving) residual block-based CNN encoder,
    e.g. Efficient-Zero (https://arxiv.org/pdf/2111.00210.pdf)"""

    def __init__(self, obs_shape, context_dim=0, out_channels=64, pixels=True,  flatten=True, num_blocks=1,
                 target_tau=None, optim_lr=None, **kwargs):
        super().__init__(obs_shape=obs_shape, context_dim=context_dim, out_channels=out_channels,
                         depth=0, num_blocks=num_blocks,
                         pixels=pixels, flatten=flatten,
                         target_tau=target_tau, optim_lr=optim_lr)

        pre_residual = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels,
                                               kernel_size=3, stride=2,
                                               padding=1, bias=False),
                                     nn.BatchNorm2d(self.out_channels - 1))

        self.CNN = nn.Sequential(Residual(pre_residual),
                                 nn.ReLU(),
                                 *[ResidualBlock(self.out_channels, self.out_channels)
                                   for _ in range(num_blocks)])

        self.init()

        # Isotropic
        assert obs_shape[-2] == self.repr_shape[1]
        assert obs_shape[-1] == self.repr_shape[2]
