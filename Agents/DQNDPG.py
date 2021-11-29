# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch

import Utils

from Blocks.encoders import CNNEncoder
from Blocks.actors import TruncatedGaussianActor, CategoricalCriticActor
from Blocks.critics import EnsembleQCritic

from Losses.QLearning import ensembleQLearning
from Losses.PolicyLearning import deepPolicyGradient


class DQNDPGAgent(torch.nn.Module):
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 target_tau, stddev_schedule, stddev_clip,  # Models
                 lr, update_per_steps,  # Optimization
                 explore_steps,  # Exploration
                 discrete, device, log_tensorboard  # On-boarding
                 ):
        super().__init__()

        self.update_per_steps = update_per_steps
        self.explore_steps = explore_steps
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
                                        optim_lr=lr).to(device)  # todo maybe don't use sched/clip as default arch

    def act(self, obs):
        with torch.no_grad(), Utils.eval_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device)
            obs = self.encoder(obs.unsqueeze(0))
            dist = self.actor(obs, self.step)
            if self.training:
                self.step += 1
                action = dist.sample()
                if self.step < self.explore_steps:
                    action = torch.randint(self.actor.action_dim, size=action.shape) if self.discrete \
                        else action.uniform_(-1, 1)
            else:
                action = dist.best if self.discrete else dist.mean
            return action.cpu().numpy()[0]

    @Utils.optimize('encoder', 'critic')
    def update_critic(self, obs, action, reward, discount, next_obs, dist=None, logs=None):
        # Critic loss
        return ensembleQLearning(self.actor, self.critic, obs, action, reward, discount, next_obs, self.step, dist,
                                 logs=logs if self.log_tensorboard else None)

    @Utils.optimize('actor')
    def update_actor(self, obs, logs=None):
        if not self.discrete:
            # Actor loss
            return deepPolicyGradient(self.actor, self.critic, obs.detach(), self.step,
                                      logs=logs if self.log_tensorboard else None)

    def update(self, replay_iter):
        logs = {'episode': self.episode, 'step': self.step}
        if self.step % self.update_per_steps != 0:
            return logs

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, *traj = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r = traj

        # Encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.log_tensorboard:
            logs['batch_reward'] = reward.mean().item()

        # Update critic
        self.update_critic(obs, action, reward, discount, next_obs,
                           logs=logs if self.log_tensorboard else None)

        # Update actor
        self.update_actor(obs,
                          logs=logs if self.log_tensorboard else None)

        # Update critic target
        self.critic.update_target_params()

        return logs
