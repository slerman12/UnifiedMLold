# Copyright Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import random
import re
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


# Saves Torch objects to root
def save(root_path, **to_save):
    save_path = root_path / 'Saved.pt'
    with save_path.open('wb') as f:
        torch.save(to_save, f)


# Loads Torch objects from root
def load(root_path, *keys):
    save_path = root_path / 'Saved.pt'
    print(f'resuming: {save_path}')
    with save_path.open('rb') as f:
        loaded = torch.load(f)
    return tuple(loaded[k] for k in keys)


# Backward pass on a loss; clear the grads of models; step their optimizers
def optimize(loss=None, *models, clear_grads=True, backward=True, step_optim=True):
    # Clear grads
    if clear_grads:
        for model in models:
            model.optim.zero_grad(set_to_none=True)

    # Backward
    if backward and loss is not None:
        loss.backward()

    # Optimize
    if step_optim:
        for model in models:
            model.optim.step()


# Context manager that temporarily switches on torch.no_grad() and eval() mode for specified models; then resets them
class act_mode:
    def __init__(self, *models):
        super().__init__()
        self.models = models

        self.with_no_grad = torch.no_grad()

    def __enter__(self):
        self.with_no_grad.__enter__()

        self.start_modes = []
        for model in self.models:
            self.start_modes.append(model.training)
            model.eval()

    def __exit__(self, *args):
        self.with_no_grad.__exit__(*args)

        for model, mode in zip(self.models, self.start_modes):
            model.train(mode)
        return False


# Sets all Torch and Numpy random seeds
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Copies parameters from one model to another
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


# Converts data to Torch Tensors and moves them to the specified device
def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device).float() for x in xs)


# Initializes model weights according to common distributions
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# A Normal distribution with its variance clipped TODO clip via differentiable re-param trick
class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6, clip=None):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.clip = clip

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    # Defaults to no clip, no grad
    def sample(self, clip=False, sample_shape=torch.Size()):
        return self.rsample(clip=clip, sample_shape=sample_shape).detach()

    # Defaults to clip, grad
    def rsample(self, clip=True, sample_shape=torch.Size()):
        clip = self.clip if clip else None
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps = eps * self.scale
        if clip is not None:
            eps_ = torch.clamp(eps, -clip, clip)  # Don't explore /too/ much
            eps[eps < -clip] = (eps + eps_ - eps.detach())[eps < -clip]  # TODO ...by multiplying scale (st.dev) /after/ clipping)
            eps[eps > clip] = (eps + eps_ - eps.detach())[eps > clip]
        x = self.loc + eps
        return self._clamp(x)


def schedule(sched, step):
    try:
        return float(sched)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', sched)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', sched)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(sched)

