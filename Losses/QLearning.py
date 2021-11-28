# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F


def ensembleQLearning(actor, critic, obs, action, reward, discount, next_obs, step, dist=None, entropy_temp=None,
                      sub_planner=None, planner=None, logs=None):
    with torch.no_grad():
        next_dist = actor(next_obs, step)
        next_action = next_dist.rsample()
        if sub_planner is not None and planner is not None:
            # todo targets for both?
            next_obs = sub_planner.target(next_obs, next_action)
            # next_obs = sub_planner.target(next_obs)  # state-action based planner
            next_obs = planner.target(next_obs)  # state-based planner
            # next_obs = torch.layer_norm(next_obs, next_obs.shape)  # todo
        next_Qs = critic.target(next_obs, next_action, next_dist)
        next_Q = torch.min(*next_Qs)
        # todo does this get grad?
        if entropy_temp is not None:
            next_log_pi = next_dist.log_prob(next_action).sum(-1, keepdim=True)
            next_Q = next_Q - entropy_temp.detach() * next_log_pi
            # todo I think this gets grad? or all factors are predefined temp=.03, scaling=.9, lo=-1
            if munch_scaling is not None and munch_lo is not None:
                # compute Munchausen_reward
                dist = actor(obs)
                log_pi = dist.log_prob(action).mean(-1, keepdim=True)
                # todo might not be right for nstep (need one munch log_pi per time step)
                reward += munch_scaling * torch.clamp(entropy_temp * log_pi, min=munch_lo, max=0)
                # todo target gets multiplied by proba?
        target_Q = reward + (discount * next_Q)

    if sub_planner is not None and planner is not None:
        obs = sub_planner(obs, action)  # state-action based planner
        # obs = sub_planner(obs)  # state-based planner
        obs = planner(obs)
        # obs = torch.layer_norm(obs, obs.shape)  # todo

    Qs = critic(obs, action, dist)
    # todo huber loss?
    bellman_error = sum([F.mse_loss(Q, target_Q) for Q in Qs])

    if logs is not None:
        assert isinstance(logs, dict)
        logs['target_q'] = target_Q.mean().item()
        logs.update({f'q{i}': Q.mean().item() for i, Q in enumerate(Qs)})
        logs['bellman_error'] = bellman_error.item()

    return bellman_error



