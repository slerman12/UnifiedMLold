import torch
from torch import nn

import Utils


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, depth=0,
                 batch_norm=False, batch_norm_last=False, l2_norm=False):
        super().__init__()

        self.MLP = nn.Sequential(
            *sum([[
                # Optional L2 norm of penultimate
                # (See: https://openreview.net/pdf?id=9xhgmsNVHu)
                # Similarly, Efficient-Zero initializes 2nd-to-last layer as all 0s  TODO
                Utils.L2Norm() if l2_norm and i == depth else nn.Identity(),
                nn.Linear(in_dim if i == 0 else hidden_dim,
                          hidden_dim if i < depth else out_dim),
                nn.BatchNorm1d(hidden_dim if i < depth or batch_norm_last else out_dim
                               ) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True) if i < depth else nn.Identity()
            ]
                for i in range(depth + 1)], [])
        )

        self.apply(Utils.weight_init)

    def forward(self, *x):
        return self.MLP(torch.cat(x, -1))


class MLPBlock(nn.Module):
    """MLP block:

    With LayerNorm

    Also optionally Batch-Norm MLP a la Efficient-Zero (https://arxiv.org/pdf/2111.00210.pdf)

    DrQV2: depth=1, ln=True, bn=False, last_bn = False
    Efficient-Zero: depth=1 for projector 0 for predictor, ln=False,
                    bn=True, bn_last=True for projector False for predictor

    Can also l2-normalize penultimate layer (https://openreview.net/pdf?id=9xhgmsNVHu)"""

    def __init__(self, in_dim, out_dim, feature_dim=512, hidden_dim=512, depth=1,
                 layer_norm=False, batch_norm=False, batch_norm_last=False, l2_norm=False,
                 target_tau=None, optim_lr=None):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(in_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh()) if layer_norm \
            else None

        in_features = feature_dim if layer_norm else in_dim

        self.MLP = MLP(in_features, out_dim, hidden_dim, depth=depth,
                       batch_norm=batch_norm, batch_norm_last=batch_norm_last, l2_norm=l2_norm)

        self.apply(Utils.weight_init)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(in_dim=in_dim, out_dim=out_dim,
                                    feature_dim=feature_dim, hidden_dim=hidden_dim, depth=depth, layer_norm=layer_norm,
                                    batch_norm=batch_norm, batch_norm_last=batch_norm_last, l2_norm=l2_norm)
            target.load_state_dict(self.state_dict())
            self.target = target

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, *x):
        h = torch.cat(x, -1)

        if self.trunk is not None:
            h = self.trunk(h)

        return self.MLP(h)
