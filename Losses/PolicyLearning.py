# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch


def deepPolicyGradient(actor, critic, obs, step, dist=None,
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

    policy_loss = -Q.mean() - .1 * dist.entropy().mean()

    if logs is not None:
        assert isinstance(logs, dict)
        logs['policy_loss'] = policy_loss.item()
        logs['action_probs'] = torch.exp(dist.log_prob(action)).mean().item()
        logs['policy_ent'] = dist.entropy().sum(dim=-1).mean().item()

    return policy_loss


def entropyMaxim(actor, obs, step, entropy_temp, dist=None, logs=None):
    if dist is None:
        dist = actor(obs, step)
    action = dist.rsample()  # todo try sampling multiple - why not? convolve with obs "scatter sample"

    log_pi = dist.log_prob(action).sum(-1, keepdim=True)

    entropy = entropy_temp.detach() * log_pi

    entropy_loss = entropy.mean()

    if logs is not None:
        assert isinstance(logs, dict)
        logs['entropy_loss'] = entropy_loss.item()
        logs['action_probs'] = torch.exp(dist.log_prob(action)).mean().item()
        logs['policy_ent'] = dist.entropy().sum(dim=-1).mean().item()

    return entropy_loss
