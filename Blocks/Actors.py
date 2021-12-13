# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn
from torch.distributions import Categorical, Normal, TransformedDistribution

import Utils


class Actor(nn.Module):
    """
    Base actor
    """

    def __init__(self, repr_shape, feature_dim, hidden_dim, action_dim, policy_norm=False,
                 target_tau=None, optim_lr=None, **kwargs):
        super().__init__()

        self.action_dim = action_dim

        repr_dim = math.prod(repr_shape)

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Pi_head = nn.Sequential(*[nn.Linear(feature_dim, hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(inplace=True),
                                       # See: https://openreview.net/pdf?id=9xhgmsNVHu
                                       Utils.L2Norm() if policy_norm else nn.Identity(),
                                       nn.Linear(hidden_dim, action_dim)])

        self.apply(Utils.weight_init)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(repr_shape=repr_shape, feature_dim=feature_dim, hidden_dim=hidden_dim,
                                    action_dim=action_dim, policy_norm=policy_norm, **kwargs)
            target.load_state_dict(self.state_dict())
            self.target = target

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, obs, step=None):
        h = self.trunk(obs)
        Q = self.Pi_head(h)
        return Q


class TruncatedGaussianActor(Actor):
    def __init__(self, repr_shape, feature_dim, hidden_dim, action_dim,
                 stddev_schedule=None, stddev_clip=None, policy_norm=True, discrete=False,
                 target_tau=None, optim_lr=None, **kwargs):  # note added kwargs todo check if ok
        dim = kwargs.get("dim", action_dim)  # (To make sure target has the same action_dim)

        num_outputs = 2 if stddev_schedule is None else 1

        super().__init__(repr_shape=repr_shape, feature_dim=feature_dim,
                         hidden_dim=hidden_dim, action_dim=dim * num_outputs, policy_norm=policy_norm,
                         stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                         target_tau=target_tau, optim_lr=optim_lr, dim=dim)

        self.discrete = discrete
        self.action_dim = dim
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

    def forward(self, obs, step=None):
        h = self.trunk(obs)

        if self.stddev_schedule is None or step is None:
            mu, log_std = self.Pi_head(h).chunk(2, dim=-1)
            std = log_std.exp()
        else:
            mu = self.Pi_head(h)
            std = Utils.schedule(self.stddev_schedule, step)
            std = torch.full_like(mu, std)

        self.raw_mu = mu
        mu = torch.tanh(mu)

        dist = Utils.TruncatedNormal(mu, std, clip=self.stddev_clip, one_hot=self.discrete)

        return dist


class DiagonalGaussianActor(Actor):
    """
    torch.distributions implementation of an diagonal Gaussian policy.
    """

    def __init__(self, repr_shape, feature_dim, hidden_dim, action_dim,
                 stddev_schedule=None, log_std_bounds=None,
                 target_tau=None, optim_lr=None, **kwargs):
        dim = kwargs.get("dim", action_dim)  # (To make sure target has the same action_dim)

        num_outputs = 2 if stddev_schedule is None else 1

        super().__init__(repr_shape=repr_shape, feature_dim=feature_dim,
                         hidden_dim=hidden_dim, action_dim=dim * num_outputs,
                         stddev_schedule=stddev_schedule, log_std_bounds=log_std_bounds,
                         optim_lr=optim_lr, target_tau=target_tau, dim=dim)

        self.tanh_transform = Utils.TanhTransform()

        self.discrete = False
        self.action_dim = dim
        self.stddev_schedule = stddev_schedule
        self.log_std_bounds = log_std_bounds

    def forward(self, obs, step=None):
        h = self.trunk(obs)

        if self.stddev_schedule is None or step is None:
            mu, log_std = self.Pi_head(h).chunk(2, dim=-1)

            # Constrain log_std inside [log_std_min, log_std_max]
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            std = log_std.exp()
        else:
            mu = self.Pi_head(h)
            std = Utils.schedule(self.stddev_schedule, step)
            std = torch.full_like(mu, std)

        dist = TransformedDistribution(Normal(mu, std), [self.tanh_transform])

        setattr(dist, 'mean', self.tanh_transform(dist.mean))

        return dist


class CategoricalCriticActor(nn.Module):
    def __init__(self, critic, stddev_schedule=None):
        super(CategoricalCriticActor, self).__init__()

        self.discrete = True
        self.stddev_schedule = stddev_schedule
        self.critic = critic
        self.action_dim = critic.action_dim

        # register module
        nn.ModuleList([critic])

        self.ensemble_size = self.critic.ensemble_size

        if hasattr(critic, "target"):
            self.target = critic.target  # TODO this doesn't work

        # TODO trainable entropy
        if hasattr(critic, "optim"):
            self.optim = critic.optim

    def forward(self, obs, step=None):
        # TODO Learnable temp
        temp = 1 if step is None else Utils.schedule(self.stddev_schedule, step)

        Qs = self.critic(obs)
        Q_min = torch.min(*Qs)  # Min-reduced
        Q = sum(*Qs) / self.ensemble_size  # Mean-reduced

        # TODO Clip with Munchausen-style lo for stability? (otherwise, exp vanishes with big logits I think)
        #  (^logits too low)
        # TODO Logits/logprob via subtracting max_q first for stability? (otherwise, exp explodes for low temps I think)
        #  (^logits too high)
        # TODO torch.logsumexp !! example of both: https://github.com/BY571/Munchausen-RL/blob/master/M-DQN.ipynb
        logits = (Q - Q.max(dim=-1, keepdim=True)[0])
        dist = Categorical(logits=logits / temp)

        # Set dist.Qs (Q1, Q2, ...)
        setattr(dist, "Qs", Qs)

        # Set dist.Q
        setattr(dist, "Q", Q)

        # Set rsample
        setattr(dist, "rsample", dist.sample)

        # Set dist.best_action
        setattr(dist, "best", torch.argmax(Q, -1))

        # Set expected Value function V
        # setattr(dist, "V", (Q_min * torch.softmax(logits / temp, dim=-1)).sum(-1))

        return dist

