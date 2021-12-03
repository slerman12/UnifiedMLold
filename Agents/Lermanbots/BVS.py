# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

import Utils

from Agents import DQNDPGAgent
from Blocks.actors import CategoricalCriticActor, TruncatedGaussianActor

from Blocks.encoders import LayerNormMLPEncoder
from Blocks.critics import EnsembleQCritic
from Blocks.planners import SubPlanner

from Losses import QLearning, PolicyLearning
from Losses.SelfSupervisedLearning import bootstrapLearningBVS


class BVSAgent(DQNDPGAgent):
    """Deep Q-Network, Deep Policy Gradient"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau, plan_discount,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, device, log_tensorboard  # On-boarding
                 ):
        super().__init__(
            obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
            lr, target_tau,  # Optimization
            explore_steps, stddev_schedule, stddev_clip,  # Exploration
            discrete, device, log_tensorboard  # On-boarding
        )

        self.plan_discount = plan_discount

        # Models
        # State based
        # self.sub_planner = LayerNormMLPEncoder(self.encoder.repr_dim, feature_dim, hidden_dim, hidden_dim,
        #                                        target_tau=target_tau, optim_lr=lr).to(device)
        self.sub_planner = LayerNormMLPEncoder(self.encoder.repr_dim, hidden_dim, hidden_dim, hidden_dim,
                                               target_tau=target_tau, optim_lr=lr).to(device)
        # State-action based
        # self.sub_planner = SubPlanner(self.encoder.repr_dim, feature_dim, hidden_dim, hidden_dim, action_shape[-1],
        #                               target_tau=target_tau, optim_lr=lr, discrete=discrete).to(device)

        self.planner = SubPlanner(hidden_dim, hidden_dim, hidden_dim, hidden_dim, action_shape[-1],
                                  target_tau=target_tau, optim_lr=lr, discrete=discrete).to(device)

        # self.planner = LayerNormMLPEncoder(hidden_dim, hidden_dim, hidden_dim, hidden_dim,
        #                                    target_tau=target_tau, optim_lr=lr).to(device)

        self.critic = EnsembleQCritic(hidden_dim, hidden_dim, hidden_dim, action_shape[-1],
                                      target_tau=target_tau, optim_lr=lr, discrete=discrete).to(device)

        self.actor = CategoricalCriticActor(self.critic, stddev_schedule) if discrete \
            else TruncatedGaussianActor(hidden_dim, hidden_dim, hidden_dim, action_shape[-1],
                                        stddev_schedule, stddev_clip,
                                        optim_lr=lr).to(device)

        # Birth

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

            # "See"
            obs = self.encoder(obs)
            obs = self.sub_planner(obs)
            # obs = self.planner(obs)
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
        traj_o[:, 0] = self.aug(traj_o[:, 0])

        # Encode
        traj_o = self.encoder(traj_o)
        obs = traj_o[:, 0]
        with torch.no_grad():
            next_obs = traj_o[:, -1].detach()

        if self.log_tensorboard:
            logs['batch_reward'] = reward.mean().item()

        # "Predict" / "Discern" / "Plan" / "Learn" / "Grow"

        # Critic loss
        critic_loss = QLearning.ensembleQLearning(self.actor, self.critic,
                                                  obs, action, reward, discount, next_obs, self.step,
                                                  sub_planner=self.sub_planner, planner=self.planner,
                                                  logs=logs)

        # Planner loss
        planner_loss = bootstrapLearningBVS(self.actor, self.sub_planner, self.planner,
                                            obs.detach(), traj_o.detach(), self.plan_discount,
                                            traj_a, self.step,  # Comment out for state-based
                                            logs=logs)

        # Update planner-critic
        Utils.optimize(planner_loss + critic_loss,
                       self.encoder,
                       self.critic,
                       self.sub_planner,
                       self.planner)

        self.critic.update_target_params()

        # self.sub_planner.update_target_params()  # Maybe not since kind of treated as encoder
        self.planner.update_target_params()

        # Actor loss
        if not self.discrete:
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(), self.step,
                                                           sub_planner=self.sub_planner, planner=self.planner,
                                                           logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor,
                           self.sub_planner)

        return logs
