import torch
from torch import nn

import Utils


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=512, depth=0, target_tau=None, optim_lr=None):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(inplace=True),
            *sum([[nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)]
                  for _ in range(depth)], []),
            nn.Linear(hidden_size, out_dim)
        )

        self.apply(Utils.weight_init)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(in_dim=in_dim, out_dim=out_dim, hidden_size=hidden_size, depth=depth)
            target.load_state_dict(self.state_dict())
            self.target = target

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, x):
        return self.mlp(x)
