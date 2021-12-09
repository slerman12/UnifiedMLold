import torch
from torch import nn
import Utils


"""Creators: As in, "to create." Generative models that plan, forecast, and imagine."""


class SubPlanner(nn.Module):
    """SubPlanner state-action based network."""
    def __init__(self, repr_dim, feature_dim, hidden_dim, output_dim, action_dim,
                 target_tau=None, optim_lr=None, discrete=False):
        super().__init__()

        self.output_dim = output_dim
        self.action_dim = action_dim
        self.discrete = discrete

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        in_dim = feature_dim if discrete else feature_dim + action_dim
        out_dim = action_dim * output_dim if discrete else output_dim

        self.P_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim))

        self.apply(Utils.weight_init)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(repr_dim=repr_dim, feature_dim=feature_dim, hidden_dim=hidden_dim,
                                    output_dim=output_dim, action_dim=action_dim, discrete=discrete)
            target.load_state_dict(self.state_dict())
            self.target = target

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, obs, action):
        h = self.trunk(obs)

        h = torch.cat([h, action], dim=-1)

        p = self.P_net(h)

        if self.discrete:
            p = p.view(*p.shape[:-1], -1, self.action_dim)
            ind = action.long().view(*p.shape[:-1], 1)
            p = torch.gather(p, -1, ind).squeeze(-1)

        return p
