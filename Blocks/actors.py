# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn
from torch.distributions import Categorical

import Utils


class BaseActor(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim, action_dim, target_tau=None, optim_lr=None, **kwargs):
        super().__init__()

        self.action_dim = action_dim

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))

        self.apply(Utils.weight_init)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(repr_dim=repr_dim, feature_dim=feature_dim,
                                    hidden_dim=hidden_dim, action_dim=action_dim, **kwargs)
            target.load_state_dict(self.state_dict())
            self.target = target

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, obs, step=None):
        h = self.trunk(obs)
        Q = self.policy(h)
        return Q


class TruncatedGaussianActor(BaseActor):
    def __init__(self, repr_dim, feature_dim, hidden_dim, action_dim,
                 stddev_schedule=None, stddev_clip=None,
                 target_tau=None, optim_lr=None):
        super().__init__(repr_dim, feature_dim, hidden_dim, action_dim,
                         target_tau=target_tau, optim_lr=optim_lr)

        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

    def forward(self, obs, step=None):
        stddev = 0 if step is None or self.stddev_schedule is None else Utils.schedule(self.stddev_schedule, step)

        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * stddev

        dist = Utils.TruncatedNormal(mu, std, clip=self.stddev_clip)

        return dist


class DiagonalGaussianActor(BaseActor):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, repr_dim, feature_dim, hidden_dim, action_dim, log_std_bounds,
                 target_tau=None, optim_lr=None, **kwargs):
        dim = kwargs.get("dim", action_dim)

        super().__init__(repr_dim=repr_dim, feature_dim=feature_dim,
                         hidden_dim=hidden_dim, action_dim=dim * 2,
                         target_tau=target_tau, optim_lr=optim_lr, dim=dim)

        self.action_dim = dim
        self.log_std_bounds = log_std_bounds

    def forward(self, obs, step=None):
        h = self.trunk(obs)

        mu, log_std = self.policy(h).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = Utils.SquashedNormal(mu, std)

        return dist


class CategoricalCriticActor(nn.Module):
    def __init__(self, critic, stddev_schedule=None):
        super(CategoricalCriticActor, self).__init__()

        self.stddev_schedule = stddev_schedule
        self.critic = critic
        self.action_dim = critic.action_dim

        # register module
        nn.ModuleList([critic])

        self.ensemble_size = self.critic.ensemble_size

        if hasattr(critic, "target"):
            self.target = critic.target

        if hasattr(critic, "optim"):
            self.optim = critic.optim

    def forward(self, obs, step=None):
        temp = 1 if step is None else Utils.schedule(self.stddev_schedule, step)

        Qs = self.critic(obs)
        # Q = torch.min(*Qs)  # min-reduced
        Q = sum(*Qs) / self.ensemble_size  # mean-reduced

        dist = Categorical(logits=Q / temp)

        # set dist.Qs (Q1, Q2, ...)
        setattr(dist, "Qs", Qs)

        # set dist.Q
        setattr(dist, "Q", Q)

        # set rsample
        setattr(dist, "rsample", dist.sample)

        # set dist.best_action
        setattr(dist, "best", torch.argmax(Q, -1))

        return dist

