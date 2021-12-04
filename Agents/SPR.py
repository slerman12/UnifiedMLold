# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch

import Utils

from Blocks.augmentations import IntensityAug, RandomShiftsAug
from Blocks.encoders import CNNEncoder, LayerNormMLPEncoder, IsotropicCNNEncoder
from Blocks.actors import TruncatedGaussianActor, CategoricalCriticActor
from Blocks.critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning, SelfSupervisedLearning


class SPRAgent(torch.nn.Module):
    """Self-Predictive Representations"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, device, log_tensorboard  # On-boarding
                 ):
        super().__init__()

        self.discrete = discrete
        self.device = device
        self.log_tensorboard = log_tensorboard
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps

        # Models
        self.encoder = CNNEncoder(obs_shape, flatten=False, target_tau=target_tau, optim_lr=lr).to(device)

        self.dynamics = IsotropicCNNEncoder(self.encoder.repr_shape, out_channels=self.encoder.out_channels,
                                            action_dim=action_shape[-1], flatten=False, optim_lr=lr).to(device)

        self.projection_g = LayerNormMLPEncoder(self.encoder.repr_dim, hidden_dim, hidden_dim, hidden_dim,
                                                target_tau=target_tau, optim_lr=lr).to(device)

        self.prediction_q = LayerNormMLPEncoder(hidden_dim, hidden_dim, hidden_dim, hidden_dim,
                                                optim_lr=lr).to(device)

        self.critic = EnsembleQCritic(self.encoder.repr_dim, feature_dim, hidden_dim, action_shape[-1],
                                      target_tau=target_tau, optim_lr=lr, discrete=discrete).to(device)

        self.actor = CategoricalCriticActor(self.critic, stddev_schedule) if discrete \
            else TruncatedGaussianActor(self.encoder.repr_dim, feature_dim, hidden_dim, action_shape[-1],
                                        stddev_schedule, stddev_clip,
                                        optim_lr=lr).to(device)

        # Data augmentation
        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

        # Birth

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

            # "See"
            obs = self.encoder(obs).flatten(-3)
            dist = self.actor(obs, self.step)

            action = dist.sample() if self.training \
                else dist.best if self.discrete \
                else dist.mean

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps and self.training:
                    action = torch.randint(self.actor.action_dim, size=action.shape) if self.discrete \
                        else action.uniform_(-1, 1)

            return action

    # "Dream"
    def update(self, replay):
        logs = {'episode': self.episode, 'step': self.step} if self.log_tensorboard \
            else None

        # "Recollect"

        batch = replay.sample()  # Can also write 'batch = next(replay)'
        obs, action, reward, discount, next_obs, *traj = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r = traj

        # "Imagine" / "Envision"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)  # TODO don't augment next

        # Encode
        obs = self.encoder(obs).flatten(-3)
        with torch.no_grad():
            next_obs = self.encoder(next_obs).flatten(-3)

        if self.log_tensorboard:
            logs['batch_reward'] = reward.mean().item()

        # "Predict" / "Discern" / "Plan" / "Learn" / "Grow"

        # Critic loss
        critic_loss = QLearning.ensembleQLearning(self.actor, self.critic,
                                                  obs, action, reward, discount, next_obs,
                                                  self.step, logs=logs)

        # Convert discrete action trajectories to one-hot
        if self.discrete:
            traj_a = Utils.one_hot(traj_a, num_classes=self.actor.action_dim)

        # Dynamics loss
        dynamics_loss = SelfSupervisedLearning.dynamicsLearning(self.dynamics, self.projection_g, self.prediction_q,
                                                                self.encoder,  # Is encoder target necessary?
                                                                traj_o, traj_a, depth=3, cheaper=True, logs=logs)

        # Update critic, dynamics
        Utils.optimize(critic_loss + dynamics_loss,  # Paper weighed dynamics loss by 2
                       self.encoder,
                       self.critic,
                       self.dynamics, self.projection_g, self.prediction_q)

        self.encoder.update_target_params()
        self.critic.update_target_params()
        self.projection_g.update_target_params()

        # Actor loss
        if not self.discrete:
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor)

        return logs
