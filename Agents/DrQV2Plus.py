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
from Blocks.Architectures.MLP import MLP

from Losses import QLearning, PolicyLearning, SelfSupervisedLearning


class DrQV2PlusAgent(torch.nn.Module):
    """Variance-Reduced Data-Regularized Q-Network (https://openreview.net/pdf?id=9xhgmsNVHu)"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, device, log_tensorboard  # On-boarding
                 ):
        super().__init__()

        # ! Original only compatible with continuous spaces, both supported here
        self.discrete = discrete  # Discrete (e.g. Atari) supported
        self.device = device
        self.log_tensorboard = log_tensorboard
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps

        # Models
        self.encoder = CNNEncoder(obs_shape, target_tau=target_tau, optim_lr=lr).to(device)

        self.critic = EnsembleQCritic(self.encoder.repr_dim, feature_dim, hidden_dim, action_shape[-1],
                                      critic_norm=True,
                                      target_tau=target_tau, optim_lr=lr, discrete=discrete).to(device)

        self.actor = CategoricalCriticActor(self.critic, stddev_schedule) if discrete \
            else TruncatedGaussianActor(self.encoder.repr_dim, feature_dim, hidden_dim, action_shape[-1],
                                        policy_norm=True,
                                        stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                        optim_lr=lr).to(device)

        self.self_supervisor = MLP(feature_dim, feature_dim, target_tau=target_tau, optim_lr=lr).to(device)

        # Data augmentation
        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

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
        obs, action, reward, discount, next_obs_orig, *traj = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r = traj

        # "Imagine" / "Envision"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs_orig)

        # Encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)  # TODO encoder target? then "concept" not needed. or no target fr CL

        if self.log_tensorboard:
            logs['batch_reward'] = reward.mean().item()

        # "Predict" / "Discern" / "Learn" / "Grow"

        # Critic loss
        critic_loss = QLearning.ensembleQLearning(self.actor, self.critic,
                                                  obs, action, reward, discount, next_obs,
                                                  self.step, ensemble_reduction='mean', logs=logs)

        # Self supervision loss
        self_supervision_loss = SelfSupervisedLearning.bootstrapYourOwnLatent(self.encoder, self.critic,
                                                                              self.self_supervisor,
                                                                              obs, next_obs_orig)

        # Update critic
        Utils.optimize(critic_loss + self_supervision_loss,
                       self.encoder,
                       self.critic,
                       self.self_supervisor)

        self.encoder.update_target_params()  # note: should Utils.optimize optionally update target params as well?
        self.critic.update_target_params()

        # Actor loss
        if not self.discrete:
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, logs=logs)

            # Update actor
            Utils.optimize(actor_loss + 0.000001 * self.actor.raw_mu.square().mean(),
                           self.actor)

        return logs
