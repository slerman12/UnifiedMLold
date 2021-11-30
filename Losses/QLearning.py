# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F


def ensembleQLearning(actor, critic, obs, action, reward, discount, next_obs, step, dist=None, entropy_temp=0,  # 0.03
                      sub_planner=None, planner=None, munchausen_scaling=0, logs=None):  # 0.9
    with torch.no_grad():
        next_dist = actor(next_obs, step)
        next_action = next_dist.rsample()

        # BVS Planning
        if sub_planner is not None and planner is not None:
            next_obs = sub_planner.target(next_obs, next_action)  # state-action-based planner TODO targets for both?
            # next_obs = sub_planner.target(next_obs)  #  state-based planner  TODO try this
            next_obs = planner.target(next_obs)
            # next_obs = torch.layer_norm(next_obs, next_obs.shape)  TODO try normalizing

        # Ensemble Q learning
        next_Q_ensemble = critic.target(next_obs, next_action, next_dist)
        next_Q = torch.min(*next_Q_ensemble)

        # Future uncertainty maximization in reward  TODO consider N-step entropy... +actor(traj_o).entropy(traj_a)
        # next_action_log_proba = next_dist.log_prob(next_action).sum(-1, keepdim=True)
        # Entropy in future decisions means exploring the uncertain, the lesser-explored
        next_entropy = next_dist.entropy().mean(-1, keepdim=True)
        # TODO each Q target gets multiplied by proba in expectation
        # target_Q = reward + (discount * next_Q - entropy_temp * next_action_log_proba)
        # TODO the above version would go well with differentiable next_dist, otherwise below is fine
        # target_Q = reward + (discount * next_Q + entropy_temp * next_entropy)
        target_Q = reward + discount * next_Q + entropy_temp * next_entropy

        # "Munchausen reward":
        # Current certainty maximization in reward, thereby increasing so-called "action-gap"
        # Furthermore, off-policy sampling of outdated rewards might be mitigated to a degree by on-policy estimate
        if munchausen_scaling != 0:
            if dist is None:
                dist = actor(obs)
            action_log_proba = dist.log_prob(action).mean(-1, keepdim=True)
            # entropy = dist.entropy().mean(-1, keepdim=True)
            lo = -1
            target_Q += munchausen_scaling * torch.clamp(entropy_temp * action_log_proba, min=lo, max=0)
            # target_Q -= munchausen_scaling * entropy  # But here Q depends on a, so I think above log_proba prefered

    # BVS Planning
    if sub_planner is not None and planner is not None:
        obs = sub_planner(obs, action)  # state-action based planner
        # obs = sub_planner(obs)  # state-based planner  TODO try this
        obs = planner(obs)
        # obs = torch.layer_norm(obs, obs.shape)  TODO try normalizing

    Q_ensemble = critic(obs, action, dist)

    # Temporal difference error (via MSE)  TODO huber loss?
    td_error = sum([F.mse_loss(Q, target_Q) for Q in Q_ensemble])

    if logs is not None:
        assert isinstance(logs, dict)
        logs['target_q'] = target_Q.mean().item()
        logs.update({f'q{i}': Q.mean().item() for i, Q in enumerate(Q_ensemble)})
        logs['td_error'] = td_error.item()

    return td_error
