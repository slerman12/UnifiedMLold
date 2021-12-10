import torch
from torch import nn

import Utils


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, depth=0,
                 batch_norm=False, batch_norm_last=False, l2_norm=False):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                                 nn.ReLU(inplace=True),
                                 *sum([[nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                                        nn.ReLU(inplace=True)]
                                       for _ in range(depth)], []),
                                 # See: https://openreview.net/pdf?id=9xhgmsNVHu
                                 Utils.L2Norm() if l2_norm else nn.Identity(),
                                 # Similarly, Efficient-Zero initializes 2nd-to-last layer as all 0s  TODO
                                 nn.Linear(hidden_dim, out_dim),
                                 nn.BatchNorm1d(hidden_dim) if batch_norm_last else nn.Identity())

        self.apply(Utils.weight_init)

    def forward(self, x):
        return self.mlp(x)


class LayerNormMLPBlock(nn.Module):
    """Layer-norm MLP block e.g., DrQV2 (https://arxiv.org/abs/2107.09645).

    Also optionally Batch-Norm MLP a la Efficient-Zero (https://arxiv.org/pdf/2111.00210.pdf)

    DrQV2: depth=1, ln=True, bn=False, last_bn = False
    Efficient-Zero: depth=1 for projector 0 for predictor, ln=False,
                    bn=True, bn_last=True for projector False for predictor

    Can also l2-normalize penultimate layer (https://openreview.net/pdf?id=9xhgmsNVHu)"""

    def __init__(self, in_dim, out_dim, feature_dim, hidden_dim, depth=1,
                 layer_norm=True, batch_norm=False, batch_norm_last=False, l2_norm=False,
                 target_tau=None, optim_lr=None):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(in_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh()) if layer_norm \
            else None

        self.net = MLP(feature_dim, out_dim, hidden_dim, depth=depth,
                       batch_norm=batch_norm, batch_norm_last=batch_norm_last, l2_norm=l2_norm)

        self.apply(Utils.weight_init)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(in_dim=in_dim, out_dim=out_dim,
                                    feature_dim=feature_dim, hidden_dim=hidden_dim, depth=depth,
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

        return self.net(h)
