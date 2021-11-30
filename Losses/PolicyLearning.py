# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch


def deepPolicyGradient(actor, critic, obs, step, entropy_temp=0, trust_region_scale=0, dist=None,
                       sub_planner=None, planner=None, logs=None):
    if dist is None:
        dist = actor(obs, step)
    action = dist.rsample()  # todo try sampling multiple - why not? convolve with obs "scatter sample"
    if sub_planner is not None and planner is not None:
        obs = sub_planner(obs, action)
        obs = planner(obs)
        # obs = torch.layer_norm(obs, obs.shape)

    Qs = critic(obs, action)
    Q = torch.min(*Qs)

    log_proba = dist.log_prob(action)

    entropy = dist.entropy().mean()  # TODO or use -log_proba.mean() ?

    trust_region = torch.kl_div(log_proba, log_proba.detach()).mean()

    policy_loss = -Q.mean() - entropy_temp * entropy + trust_region_scale * trust_region

    if logs is not None:
        assert isinstance(logs, dict)
        logs['policy_loss'] = policy_loss.item()
        logs['avg_action_proba'] = torch.exp(dist.log_prob(action)).mean().item()
        logs['avg_policy_entropy'] = entropy.item()
        logs['avg_trust_region'] = trust_region.item()

    # TODO DEBUGGING delete
    if step % 1000 == 0:
        print('avg action proba', logs['action_proba'])

    return policy_loss
