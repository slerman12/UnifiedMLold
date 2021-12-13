# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch

import Utils

from Blocks.Augmentations import IntensityAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic
from Blocks.Architectures.Lermanblocks.AGIGradient import AGIGradient

from Losses import PolicyLearning, AGIQLearning


class AGIAgent(torch.nn.Module):
    """Deep Q-Network, Deep Policy Gradient"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 device, log_tensorboard, load=False,  # On-boarding
                 **kwargs):
        super().__init__()

        self.discrete = True
        self.device = device
        self.log_tensorboard = log_tensorboard
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps

        # Encoder
        self.encoder = CNNEncoder(obs_shape, optim_lr=lr).to(device)

        # Critic
        args = dict(repr_shape=self.encoder.repr_shape,
                    feature_dim=feature_dim, hidden_dim=hidden_dim,
                    discrete=True, action_dim=action_shape[-1], ensemble_size=2)
        self.critic = EnsembleQCritic(**args, target_tau=target_tau).to(device)

        # AGI Gradient as critic Q ensemble
        self.critic.Q_head = torch.nn.ModuleList([AGIGradient(in_dim=args['feature_dim'],
                                                              out_dim=args['action_dim'], depth=6,
                                                              steps=2, meta_learn_steps=512,
                                                              num_dists=32, num_samples=32,
                                                              forget_proba=0.1, teleport_proba=0.1,
                                                              optim_lr=0.001,  target_tau=target_tau,
                                                              device=device)
                                                  for _ in range(args['ensemble_size'])])
        self.critic.trunk[1] = self.critic.target.trunk[1] = Utils.L2Norm()
        self.critic.trunk[2] = self.critic.target.trunk[2] = torch.nn.Identity()
        for i in range(args['ensemble_size']):
            self.critic.target.Q_head[i] = self.critic.Q_head.target[i]
        self.critic.__post__(**args, optim_lr=lr)
        print(self.critic)

        # Critic as actor
        self.actor = CategoricalCriticActor(self.critic, stddev_schedule)

        # Data augmentation
        self.aug = IntensityAug(0.05)

        # Birth

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

            # "See"
            obs = self.encoder(obs)
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
        next_obs = self.aug(next_obs)

        # Encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.log_tensorboard:
            logs['batch_reward'] = reward.mean().item()

        # "Predict" / "Discern" / "Learn" / "Grow"

        # Critic loss
        critic_loss = AGIQLearning.ensembleQLearning(self.actor, self.critic,
                                                     obs, action, reward, discount, next_obs,
                                                     self.step, meta_learn=True, logs=logs)

        # Update critic
        Utils.optimize(critic_loss,
                       self.encoder,
                       self.critic)

        self.critic.update_target_params()
        for Q_head in self.critic.Q_head:
            Q_head.update_target()

        # Actor loss
        if not self.discrete:
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor)

        return logs
