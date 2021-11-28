# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn
import Utils


class EnsembleQCritic(nn.Module):
    """Critic network, employs ensemble Q learning."""
    def __init__(self, repr_dim, feature_dim, hidden_dim, action_dim, ensemble_size=2,
                 target_tau=None, optim_lr=None, discrete=False, **kwargs):
        super().__init__()

        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.discrete = discrete

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        in_dim = feature_dim if discrete else feature_dim + action_dim
        Q_dim = action_dim if discrete else 1

        self.Q_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True), nn.Linear(hidden_dim, Q_dim)
            )
            for _ in range(ensemble_size)])

        self.apply(Utils.weight_init)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(repr_dim=repr_dim, feature_dim=feature_dim, hidden_dim=hidden_dim,
                                    action_dim=action_dim, ensemble_size=ensemble_size, discrete=discrete, **kwargs)
            target.load_state_dict(self.state_dict())
            self.target = target

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, obs=None, action=None, dist=None):
        if self.discrete:
            assert obs is not None or dist is not None
        else:
            assert obs is not None and action is not None
            dist = None

        if dist is None:
            h = self.trunk(obs)
            if not self.discrete:
                h = torch.cat([h, action], dim=-1)
            # get Q1, Q2, ...
            Qs = tuple(Q_net(h) for Q_net in self.Q_nets)
        else:
            # get dist.Q1, dist.Q2, ...
            Qs = dist.Qs
        if self.discrete and action is not None:  # analogous to dist.log_probs(action) except w.r.t. Qs, not logits
            # get Q1[action], Q2[action], ...
            ind = action.long().view(*Qs[0].shape[:-1], 1)
            Qs = tuple(torch.gather(Q, -1, ind) for Q in Qs)
        return Qs


class EnsemblePROCritic(EnsembleQCritic):
    def __init__(self, repr_dim, feature_dim, hidden_dim, action_dim=0, ensemble_size=2,
                 target_tau=None, optim_lr=None, discrete=False, **kwargs):

        size = kwargs.get("size", ensemble_size)  # so that self and target have consistent ensemble sizes
        super().__init__(repr_dim, feature_dim, hidden_dim, action_dim if discrete else 0, size * 2,
                         target_tau, optim_lr, discrete, size=size)

        self.action_dim = action_dim
        self.ensemble_size = size

    def forward(self, obs=None, action=None, dist=None):
        assert obs is not None and dist is not None

        h = self.trunk(obs)

        AV = tuple(mlp(h) for mlp in self.Q_nets)

        A = AV[:self.ensemble_size]
        V = AV[self.ensemble_size:]

        if self.discrete:
            log_pi = dist.logits

            # get Q1, Q2, ...
            Qs = tuple((torch.abs(a) * log_pi + v) for a, v in zip(A, V))

            if action is not None:
                # get Q1[action], Q2[action], ...
                ind = action.long().view(*Qs[0].shape[:-1], 1)
                Qs = tuple(torch.gather(Q, -1, ind) for Q in Qs)
        else:
            # todo consider scatter sampling, subtracting or adding |m|*mean, w/ or w/o b,
            #  "dueling" Q=V(s)+A(s,a)-A(s).mean
            log_pi = dist.log_prob(action)

            # get Q1, Q2, ...
            Qs = tuple((torch.abs(a) * log_pi + v).mean(-1, keepdim=True) for a, v in zip(A, V))

        return Qs
