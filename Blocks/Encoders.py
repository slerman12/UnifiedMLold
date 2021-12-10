# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

import Utils

from Blocks.Architectures.Residual import ResidualBlock, Residual


class _CNNEncoder(nn.Module):
    """Base CNN encoder."""
    def __init__(self):
        super().__init__()
        self.CNN = None
        self.neck = nn.Identity()

    def __post__(self, obs_shape, out_channels, optim_lr=None, target_tau=None, **kwargs):
        assert self.CNN is not None, 'Inheritor of Encoder must define self.CNN'

        # Initialize weights
        self.apply(Utils.weight_init)

        # Optimizer
        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        # EMA
        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(obs_shape=obs_shape, out_channels=out_channels, **kwargs)
            target.load_state_dict(self.state_dict())
            self.target = target

        # CNN feature map sizes
        self.obs_shape = obs_shape
        in_channels, height, width = obs_shape
        height, width = Utils.cnn_output_shape(height, width, self.CNN)
        self.repr_shape = (out_channels, height, width)  # Feature map shape
        self.repr_dim = math.prod(self.repr_shape)  # Flattened features dim

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    # Encodes
    def forward(self, obs, context=None):
        shape = obs.shape  # Preserve leading dims
        obs = obs.reshape(-1, *self.obs_shape)  # Encode last 3 dims 

        # Normalizes pixels
        if self.pixels:
            obs = obs / 255.0 - 0.5

        # Optionally append context to channels assuming dimensions allow
        if context is not None:
            assert len(context.shape) == 2 and \
                    self.in_channels == obs.shape[1] + context.shape[1]

            action = context.reshape(-1, context.shape[-1])[:, :, None, None].expand(-1, -1, *obs.shape[-2:])
            obs = torch.cat([obs,  action], 1)

        # CNN encode
        h = self.CNN(obs)

        # Optionally flatten output
        if self.flatten:
            h = h.view(*shape[:-3], -1)
            assert h.shape[-1] == self.repr_dim
        else:
            h = h.view(*shape[:-3], *h.shape[-3:])
            assert tuple(h.shape[-3:]) == self.repr_shape

        return self.neck(h)


class CNNEncoder(_CNNEncoder):
    """Basic CNN encoder, e.g., DrQV2 (https://arxiv.org/abs/2107.09645)."""

    def __init__(self, obs_shape, out_channels=32, depth=3, pixels=True, flatten=True,
                 optim_lr=None, target_tau=None):

        super().__init__()

        assert len(obs_shape) == 3, 'image observation shape must have 3 dimensions'

        # Dimensions
        in_channels = obs_shape[0]

        # CNN
        self.CNN = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=2), nn.ReLU(),
                                 *sum([(nn.Conv2d(out_channels, out_channels, 3, stride=1), nn.ReLU())
                                       for _ in range(depth)], ()))

        self.pixels = pixels
        self.flatten = flatten

        self.__post__(obs_shape=obs_shape, out_channels=out_channels, depth=depth,
                      pixels=pixels, flatten=flatten, optim_lr=optim_lr, target_tau=target_tau)


class ResidualBlockEncoder(_CNNEncoder):
    """Residual block-based CNN encoder, e.g., Efficient-Zero (https://arxiv.org/pdf/2111.00210.pdf)."""

    def __init__(self, obs_shape, out_channels=64, pixels=True, flatten=True, num_blocks=1,
                 optim_lr=None, target_tau=None):

        super().__init__()

        assert len(obs_shape) == 3, 'image observation shape must have 3 dimensions'

        # Dimensions
        in_channels = obs_shape[0]

        # CNN
        self.CNN = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                           kernel_size=3, padding=1, bias=False),
                                 nn.BatchNorm2d(out_channels // 2),
                                 nn.ReLU(),
                                 *[ResidualBlock(out_channels, out_channels)
                                   for _ in range(num_blocks)])

        self.pixels = pixels
        self.flatten = flatten

        self.__post__(obs_shape=obs_shape, out_channels=out_channels, num_blocks=num_blocks,
                      pixels=pixels, flatten=flatten, target_tau=target_tau, optim_lr=optim_lr)


"""
Creators: As in, "to create." 

Generative models that plan, forecast, and imagine.
"""


class IsotropicCNNEncoder(_CNNEncoder):
    """Isotropic (no bottleneck / dimensionality conserving) CNN encoder,
    e.g., SPR (https://arxiv.org/pdf/2007.05929.pdf)."""

    def __init__(self, obs_shape, context_dim=0, out_channels=None, depth=0, pixels=False, flatten=False,
                 optim_lr=None, target_tau=None):

        super().__init__()

        assert len(obs_shape) == 3, 'image observation shape must have 3 dimensions'

        # Dimensions
        self.in_channels = obs_shape[0] + context_dim
        out_channels = self.in_channels if out_channels is None else out_channels

        # CNN
        self.CNN = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, (3, 3), padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 *sum([(nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU())
                                       for _ in range(depth)], ()),
                                 nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
                                 nn.ReLU())

        self.pixels = pixels
        self.flatten = flatten

        self.__post__(obs_shape=obs_shape, context_dim=context_dim, out_channels=out_channels, depth=depth,
                      pixels=pixels, flatten=flatten, optim_lr=optim_lr, target_tau=target_tau)

        # Isotropic
        assert obs_shape[-2] == self.repr_shape[1]
        assert obs_shape[-1] == self.repr_shape[2]


class IsotropicResidualBlockEncoder(_CNNEncoder):
    """Isotropic (no bottleneck / dimensionality conserving) residual block-based CNN encoder,
    e.g. Efficient-Zero (https://arxiv.org/pdf/2111.00210.pdf)"""

    def __init__(self, obs_shape, context_dim=0, out_channels=None, num_blocks=1, pixels=False, flatten=False,
                 optim_lr=None, target_tau=None):
        super().__init__()

        assert len(obs_shape) == 3, 'image observation shape must have 3 dimensions'

        # Dimensions
        self.in_channels = obs_shape[0] + context_dim
        out_channels = self.in_channels if out_channels is None else out_channels

        # CNN
        pre_residual = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, 3, 2, 1, bias=False),
                                     nn.BatchNorm2d(out_channels - 1))

        # CNN
        self.CNN = nn.Sequential(Residual(pre_residual),
                                 nn.ReLU(),
                                 *[ResidualBlock(out_channels, out_channels)
                                   for _ in range(num_blocks)])

        self.pixels = pixels
        self.flatten = flatten

        self.__post__(obs_shape=obs_shape, context_dim=context_dim, out_channels=out_channels, num_blocks=num_blocks,
                      pixels=pixels, flatten=flatten, optim_lr=optim_lr, target_tau=target_tau)

        # Isotropic
        assert obs_shape[-2] == self.repr_shape[1]
        assert obs_shape[-1] == self.repr_shape[2]
