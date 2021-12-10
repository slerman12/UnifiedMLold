# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import random
import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


# Sets all Torch and Numpy random seeds
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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


# Copies parameters from one model to another
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


# Basic L2 normalization
class L2Norm(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, dim=-1, eps=self.eps)


# Context manager that temporarily switches on eval() mode for specified models; then resets them
class act_mode:
    def __init__(self, *models):
        super().__init__()
        self.models = models

        # self.with_no_grad = torch.no_grad()

    def __enter__(self):
        # self.with_no_grad.__enter__()

        self.start_modes = []
        for model in self.models:
            self.start_modes.append(model.training)
            model.eval()

    def __exit__(self, *args):
        # self.with_no_grad.__exit__(*args)

        for model, mode in zip(self.models, self.start_modes):
            model.train(mode)
        return False


# Converts data to Torch Tensors and moves them to the specified device as floats
def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device).float() for x in xs)


# Backward pass on a loss; clear the grads of models; step their optimizers
def optimize(loss=None, *models, clear_grads=True, backward=True, step_optim=True):
    # Clear grads  TODO (Optionally) clear grads at end instead to allow accumulation?
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

    # for model in models:
    #     # Optimize
    #     if step_optim:
    #         model.optim.step()
    #
    #     # Update EMA targets
    #     if update_targets and hasattr(model, 'target'):
    #         model.update_target_params()

    # # Update EMA targets
    # if update_targets:
    #     for model in models:
    #         if hasattr(model, 'target'):
    #             model.update_target_params()


# Increment/decrement a value in proportion to a step count based on a string-formatted schedule
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


# A Normal distribution with its variance clipped
class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6, clip=None):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.clip = clip

    def _clamp(self, x, min, max):
        clamped_x = torch.clamp(x, min, max)
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
            eps = torch.clamp(eps, -clip, clip)  # Don't explore /too/ much
            # eps = self._clamp(eps, -clip, clip)  # Don't explore /too/ much
        x = self.loc + eps

        return self._clamp(x, self.low + self.eps, self.high - self.eps)


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
        # One should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


# Compute the output shape of a CNN layer
def conv_output_shape(in_height, in_width, kernel_size=1, stride=1, padding=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    out_height = math.floor(((in_height + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    out_width = math.floor(((in_width + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return out_height, out_width


# Compute the output shape of a whole CNN
def cnn_output_shape(height, width, block):
    if isinstance(block, (nn.Conv2d, nn.AvgPool2d)):
        height, width = conv_output_shape(height, width,
                                          kernel_size=block.kernel_size,
                                          stride=block.stride,
                                          padding=block.padding)
    elif hasattr(block, 'output_shape'):
        height, width = block.output_shape(height, width)
    elif hasattr(block, 'modules'):
        for module in block.modules():
            height, width = cnn_output_shape(height, width, module)

    output_shape = (height, width)  # TODO should probably do (width, height) universally

    return output_shape


# (Multi-dim) one-hot encoding
def one_hot(x, num_classes):
    x = x.long()
    scatter_dim = len(x.shape)
    inds = x.view(*x.shape, -1)
    zeros = torch.zeros(*x.shape, num_classes, dtype=x.dtype, device=x.device)
    return zeros.scatter(scatter_dim, inds, 1)


# Converts an agent to a classifier
def to_classifier(agent):
    assert agent.discrete, "Only agents initialized as discrete\n" \
                           "can be converted to classifiers.\n" \
                           "Simply re-initialize your agent with\n" \
                           "the 'discrete' hyper-parameter set to True;\n" \
                           "all agents support discrete actions.\n"

    def update(replay):

        if agent.training:
            agent.step += 1

        # "Recollect"

        batch = replay.sample()  # Can also write 'batch = next(replay)'
        obs, y_label = to_torch(batch, agent.device)

        # "Imagine" / "Envision"

        # Augment
        if agent.training and hasattr(agent, 'aug'):
            obs = agent.aug(obs)

        # Encode
        obs = agent.encoder(obs)

        # "Predict" / "Learn" / "Grow"

        y_pred = agent.actor(obs).probs
        loss = nn.CrossEntropyLoss()(y_pred, y_label)

        # Update critic
        if agent.training:
            optimize(loss, agent.encoder, agent.critic)

        logs = {'step': agent.step,
                'loss': loss.item(),
                'accuracy': torch.sum(torch.argmax(y_pred, -1)
                                      == y_label, -1) / y_pred.shape[0]}

        return logs

    setattr(agent, 'original_update', agent.update)
    setattr(agent, 'update', update)

    return agent


# Converts a classifier to an agent
def to_agent(classifier):

    if hasattr(classifier, 'original_update'):
        update = getattr(classifier, 'original_update')
        setattr(classifier, 'update', update)

    return classifier
