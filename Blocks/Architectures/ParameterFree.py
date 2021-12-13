import math

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

import Utils


class ParameterFreeLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device='cpu') -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.Tensor(out_features, in_features).to(device)
        self.bias = torch.Tensor(out_features).to(device) if bias else None

        self.reset_tensors()

    def reset_tensors(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ParameterFreeMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, depth=0, l2_norm_1st=False, l2_norm_pen=False, device='cpu'):
        super().__init__()

        self.MLP = nn.Sequential(
            *sum([[
                # Optional L2 norm of input or penultimate
                # (See: https://openreview.net/pdf?id=9xhgmsNVHu)
                Utils.L2Norm() if (l2_norm_1st and i == 0) or
                                  (l2_norm_pen and i == depth) else nn.Identity(),
                ParameterFreeLinear(in_dim if i == 0 else hidden_dim,
                                    hidden_dim if i < depth else out_dim,
                                    device=device),
                nn.ReLU(inplace=True) if i < depth else nn.Identity()
            ]
                for i in range(depth + 1)], [])
        )

    def reset_tensors(self):
        for module in self.children():
            if isinstance(module, ParameterFreeLinear):
                module.reset_tensors()

    def forward(self, *x):
        return self.MLP(torch.cat(x, -1))
