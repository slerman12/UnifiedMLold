# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from pathlib import Path

from tqdm import tqdm

from numpy import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

import Utils

from Logger import Logger

from Blocks.Architectures.MLP import MLP
from Blocks.Architectures.ParameterFree import ParameterFreeMLP


class AGIGradient(nn.Module):
    def __init__(self, in_dim, out_dim, feature_dim=512, memory_dim=512, depth=0,
                 steps=0, meta_learn_steps=0, num_dists=0, num_samples=0, forget_proba=0., teleport_proba=0.,
                 target_tau=None, optim_lr=0.001, device='cuda'):

        super().__init__()

        path = Path.cwd()

        if (path / 'Saved.pt').exists() and False:
            self.nerves, self.hippocampus, self.crown = Utils.load(
                path, 'nerves', 'hippocampus', 'crown')
        else:
            print('Saved checkpoint not found\n'
                  'Initializing new AGI...\n'
                  'This could take a while...')

            logger = Logger(path)

            # AGI
            # self.eyes = CNN()

            self.nerves = MLP(in_dim + out_dim, feature_dim, feature_dim, depth // 3).to(device)
            self.hippocampus = nn.LSTM(feature_dim, memory_dim, depth // 3, batch_first=True).to(device)
            self.crown = MLP(in_dim + memory_dim, out_dim, memory_dim // 2, depth // 3).to(device)

            self.num_dists = num_dists

            self.null_memory = torch.zeros(depth // 3, 1, memory_dim).to(device)
            null_memory = (self.null_memory, self.null_memory)
            self.null_label = torch.zeros(1, out_dim).to(device)

            # Initial body weights
            self.apply(Utils.weight_init)

            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

            # "Batches" consist of distributions, which each generate x,y_label samples
            # AGI has a unique memory state (RNN hidden state) w.r.t. each distribution

            class Distribution:
                def __init__(self, stddev=1):
                    self.MLP = ParameterFreeMLP(in_dim, out_dim, hidden_dim=64, depth=0, device=device)
                    self.stddev = stddev

                def reset(self):
                    self.MLP.reset_tensors()

                def __call__(self, num_samples):
                    with torch.no_grad():
                        x = torch.rand(num_samples, in_dim).to(device)
                        x = F.normalize(x)
                        mu = self.MLP(x)
                        dist = Normal(mu, self.stddev)
                        # y_label = dist.sample()
                        y_label = dist.mean
                        return x, y_label

            self.distributions = [Distribution() for _ in range(num_dists)]
            self.memories = [null_memory for _ in range(num_dists)]

            for step in tqdm(range(steps), desc='Initializing AGI...'):

                x, y_label = zip(*[dist(num_samples) for dist in self.distributions])

                if step and step % meta_learn_steps == 0:
                    y_pred = self.AGI(x)

                    y_pred, y_label = map(torch.cat, [y_pred, y_label])

                    loss = F.mse_loss(y_pred, y_label)

                    logger.log({'step': step,
                                'updates': step / meta_learn_steps - 1,
                                'loss': loss.data}, dump=True)

                    Utils.optimize(loss, self)

                    self.memories = self.memories_detach()
                else:
                    self.AGI(x, label=y_label)

                # Randomly switch distributions
                if random.rand() < teleport_proba:
                    teleport_ind = random.randint(num_dists)
                    self.distributions[teleport_ind].reset()

                # Randomly forget a memory
                if random.rand() < forget_proba:
                    forget_ind = random.randint(num_dists)
                    self.memories[forget_ind] = null_memory

            self.optim.zero_grad(set_to_none=True)
            self.memories = [null_memory for _ in range(num_dists)]

            print('Initialized.\n'
                  'Saving...')

            # Save
            Utils.save(path,
                       nerves=self.nerves,
                       hippocampus=self.hippocampus,
                       crown=self.crown)

            print("Saved.")

        # Optimizer (if parametric training is desired)
        if not hasattr(self, 'optim') and optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        # EMA (w.r.t. memories rather than parameters)
        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(in_dim=in_dim, out_dim=out_dim,
                                    feature_dim=feature_dim, memory_dim=memory_dim, depth=depth)
            target.load_state_dict(self.state_dict())
            target.memories = self.memories
            self.target = target

    def update_target(self):
        assert self.target_tau is not None
        self.target.memories = [tuple(self.target_tau * self.memories[i][j]
                                      + (1 - self.target_tau) * self.target.memories[i][j]
                                      for j in [0, 1])
                                for i in range(len(self.memories))]

    def AGI(self, senses, label=None):
        update_memory = self.training and label is not None

        if label is None:
            label = [self.null_label.expand(sense.shape[0], -1) for sense in senses]

        transmits = []
        for ith, sense in enumerate(senses):
            mem = self.memories[ith]

            sense_size = sense.shape[0]
            mem_size = mem[0].shape[1]

            if sense_size < mem_size:
                mem = tuple(m[:, :sense_size].contiguous() for m in mem)
            elif sense_size > mem_size:
                mem = tuple(m.repeat(1, sense_size // mem_size, 1) for m in mem)
                nulls = self.null_memory.repeat(1, sense_size % mem_size, 1)
                mem = tuple(torch.cat([m, nulls], 1) for m in mem)

            # sight = self.eyes(sense)

            thought = self.nerves(sense, label[ith])
            recollection, mem = self.hippocampus(thought.unsqueeze(1), mem)
            if update_memory:
                self.memories[ith] = mem
            transmits.append(self.crown(sense, recollection.squeeze(1)))

        return transmits

    def forward(self, sense, label=torch.empty(0)):
        # TODO if label, do batch one at a time
        assert isinstance(sense, torch.Tensor) and isinstance(label, torch.Tensor)

        # Learns from batch, backprops sequentially
        if len(label) > 0:
            transmit = torch.cat([self.AGI((sense[i].unsqueeze(0),),
                                           (label[i].unsqueeze(0),))[0]
                                  for i in range(sense.shape[0])])
            self.memories = self.memories_detach()
        else:
            transmit = self.AGI((sense,))[0]

        # Learns from batch, backprops through last
        # if len(label) > 0:
        #     transmit = []
        #     for i in range(sense.shape[0]):
        #         transmit.append(self.AGI((sense[i].unsqueeze(0),), (label[i].unsqueeze(0),))[0])
        #         self.memories = self.memories_detach()
        #     transmit = torch.cat(transmit)
        # else:
        #     transmit = self.AGI((sense,))[0]

        # Backprops through all, learns from first
        # transmit = self.AGI((sense,), (label,) if len(label) > 0 else None)[0]
        # self.memories = self.memories_detach()

        return transmit

    def memories_detach(self):
        return [tuple(m.detach() for m in mem) for mem in self.memories]

# AGIGradient(in_dim=10, out_dim=1, depth=6,
#             steps=100000, meta_learn_steps=16,
#             num_dists=2, num_samples=2,
#             forget_proba=0.1, teleport_proba=0.1,
#             optim_lr=0.001)

# # Capacity
# AGIGradient(in_dim=10, out_dim=1, depth=18,
#             steps=100000, meta_learn_steps=524,
#             num_dists=32, num_samples=32,
#             forget_proba=0.1, teleport_proba=0.1,
#             optim_lr=0.001)
#
#
# # Initial contextualization
# AGIGradient(in_dim=10, out_dim=1, depth=6,
#             steps=100000, meta_learn_steps=16,
#             num_dists=2, num_samples=2,
#             forget_proba=0, teleport_proba=0,
#             optim_lr=0.001)
#
# # Adaptive contextualization via memorization
# AGIGradient(in_dim=10, out_dim=1, depth=6,
#             steps=100000, meta_learn_steps=16,
#             num_dists=2, num_samples=2,
#             forget_proba=0.1, teleport_proba=0,
#             optim_lr=0.001)
#
# # Adaptive contextualization via generalization
# AGIGradient(in_dim=10, out_dim=1, depth=6,
#             steps=100000, meta_learn_steps=16,
#             num_dists=2, num_samples=2,
#             forget_proba=0, teleport_proba=0.1,
#             optim_lr=0.001)
#
# # Temporal exploitation
# AGIGradient(in_dim=10, out_dim=1, depth=6,
#             steps=100000, meta_learn_steps=524,
#             num_dists=2, num_samples=2,
#             forget_proba=0.1, teleport_proba=0.1,
#             optim_lr=0.001)
#
# # Spatial exploitation
# AGIGradient(in_dim=10, out_dim=1, depth=6,
#             steps=100000, meta_learn_steps=524,
#             num_dists=2, num_samples=32,
#             forget_proba=0.1, teleport_proba=0.1,
#             optim_lr=0.001)
#
# # Breadth
# AGIGradient(in_dim=10, out_dim=1, depth=6,
#             steps=100000, meta_learn_steps=524,
#             num_dists=32, num_samples=2,
#             forget_proba=0.1, teleport_proba=0.1,
#             optim_lr=0.001)

# Distributional challenge TODO

# Continual learning via random sampling of old distributions (MNIST and CIFAR-10) TODO
# Maybe CL via CartPole, classification for above
# CNN (MNIST and CIFAR-10) TODO

# MNIST generalization? CartPole?  (vertical axes left and right, loss and reward, CartPole line has cart cartoon)
# Would also need horizontal step / episode
# Mountain car?

# 1. Synthetic via classification 18-way "to determine best"
# 2. RL Atari discrete 18-way
# 3. MNIST, Cifar-10 CNN encoding
