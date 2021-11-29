# Copyright Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import random
import re
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# Backward pass on a class method's output; clear the grads of specified models; step their optimizers
def optimize(*models):
    def decorator(method):

        @wraps(method)
        def model_loss(self, *args, clear_grads=True, step_optim=True, **kwargs):
            # Loss
            loss = method(self, *args, **kwargs)

            # Optimize
            if loss is not None:
                # Clear grads
                if clear_grads:
                    map(lambda model: getattr(self, model).optim.zero_grad(set_to_none=True), models)

                # Update models
                loss.backward()
                if step_optim:
                    map(lambda model: getattr(self, model).optim.step(), models)

        return model_loss
    return decorator


# Divides a method's batch inputs into a specified number of batch chunks and iterates the method on each chunk
# This can be used for handling large batches with less compute
def loop_in_chunks(num_chunks=1, grad_accumulation=False):
    def decorator(method):

        @wraps(method)
        def chunker(*args, **kwargs):
            chunked_args = []
            chunked_kwargs = {}
            batch_size = None
            ind = chunk_size = 0
            done = False

            # Iterate per chunk
            while not done:
                # Chunk args
                for arg in args:
                    if torch.is_tensor(arg) and hasattr(arg, 'shape'):
                        size = arg.shape[0]
                        if batch_size is None:
                            batch_size = size
                            chunk_size = batch_size // num_chunks
                            assert chunk_size > 0
                        if size == batch_size:
                            chunked_args.append(arg[ind:ind + chunk_size])
                        else:
                            chunked_args.append(arg)
                    else:
                        chunked_args.append(arg)
                # Chunk kwargs
                for k in kwargs:
                    if torch.is_tensor(kwargs[k]) and hasattr(kwargs[k], 'shape'):
                        size = kwargs[k].shape[0]
                        if batch_size is None:
                            batch_size = size
                            chunk_size = batch_size // num_chunks
                            assert chunk_size > 0
                        if size == batch_size:
                            chunked_kwargs[k] = kwargs[k][ind:ind + chunk_size]
                        else:
                            chunked_kwargs[k] = kwargs[k]
                    else:
                        chunked_kwargs[k] = kwargs[k]

                ind += chunk_size
                done = batch_size is None or ind + chunk_size >= batch_size
                # Call method on chunk
                if grad_accumulation:  # Optionally accumulates gradients if combined with @Utils.optimize decorator
                    method(*chunked_args, clear_grads=ind == 0, step_optim=done, **chunked_kwargs)
                else:
                    method(*chunked_args, **chunked_kwargs)

        return chunker
    return decorator


# No grads; temporarily switches on eval() mode for a class method's specified models; then resets them
def act_mode(*models):
    def decorator(method):

        @wraps(method)
        def set_reset_eval(self, *args, **kwargs):
            start_modes = []

            for model in models:
                start_modes.append(model.training)
                getattr(self, model).eval()

            with torch.no_grad():
                output = method(self, *args, **kwargs)

            for model, mode in zip(models, start_modes):
                getattr(self, model).train(mode)

            return output
        return set_reset_eval

    return decorator


# Context manager that temporarily switches on eval() mode for specified models; then resets them
class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.start_modes = []
        for model in self.models:
            self.start_modes.append(model.training)
            model.eval()

    def __exit__(self, *args):
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

    # defaults to no clip, no grad
    def sample(self, clip=False, sample_shape=torch.Size()):
        return self.rsample(clip=clip, sample_shape=sample_shape).detach()

    # defaults to clip, grad
    def rsample(self, clip=True, sample_shape=torch.Size()):
        clip = self.clip if clip else None
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps = eps * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
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
    raise NotImplementedError(schdl)

