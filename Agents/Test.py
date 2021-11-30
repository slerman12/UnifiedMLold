# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch

import Utils
from Blocks.augmentations import IntensityAug, RandomShiftsAug

from Blocks.encoders import CNNEncoder
from Blocks.actors import TruncatedGaussianActor, CategoricalCriticActor
from Blocks.critics import EnsembleQCritic

from Losses.QLearning import ensembleQLearning
from Losses.PolicyLearning import deepPolicyGradient


class Test(torch.nn.Module):
    """Deep Q-Network, Deep Policy Gradient"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 stddev_schedule, stddev_clip,  # Exploration
                 discrete, device, log_tensorboard  # On-boarding
                 ):
        super().__init__()

        self.discrete = discrete
        self.device = device
        self.log_tensorboard = log_tensorboard
        self.birthday = time.time()
        self.step = self.episode = 0

        # Models
        self.encoder = CNNEncoder(obs_shape, optim_lr=lr).to(device)

        self.critic = EnsembleQCritic(self.encoder.repr_dim, feature_dim, hidden_dim, action_shape[-1],
                                      target_tau=target_tau, optim_lr=lr, discrete=discrete).to(device)

        self.actor = CategoricalCriticActor(self.critic, stddev_schedule) if discrete \
            else TruncatedGaussianActor(self.encoder.repr_dim, feature_dim, hidden_dim, action_shape[-1],
                                        stddev_schedule, stddev_clip,
                                        optim_lr=lr).to(device)

        # Data augmentation
        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device)
            obs = self.encoder(obs.unsqueeze(0))
            dist = self.actor(obs, self.step)
            if self.training:
                action = dist.sample()
                if self.step < self.num_expl_steps:
                    action = torch.randint(self.actor.action_dim, size=action.shape) if self.discrete \
                        else action.uniform_(-1, 1)
            else:
                action = dist.best if self.discrete else dist.mean
            return action.cpu().numpy()[0]

    def update(self, replay):
        logs = {'episode': self.episode, 'step': self.step} if self.log_tensorboard \
            else None

        batch = replay.sample()  # Can also write 'batch = next(replay)'
        obs, action, reward, discount, next_obs, *traj = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r = traj

        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # "See"
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.log_tensorboard:
            logs['batch_reward'] = reward.mean().item()

        # Critic loss
        critic_loss = ensembleQLearning(self.actor, self.critic,
                                        obs, action, reward, discount, next_obs,
                                        self.step, logs=logs)

        # Update critic
        Utils.optimize(critic_loss,
                       self.encoder,
                       self.critic)

        self.critic.update_target_params()

        # Actor loss
        actor_loss = deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                        self.step, logs=logs)

        # Update actor
        Utils.optimize(actor_loss,
                       self.actor)

        return logs
