# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch


def deepPolicyGradient(actor, critic, obs, step, entropy_temp=0, dist=None, logs=None):
    if dist is None:
        dist = actor(obs, step)

    action = dist.mean
    # Would be great to test effect of mean vs sample
    # action = dist.rsample()  # Traditional way is to sample, but only necessary if learnable entropy/stddev, right?
    # num_actions = 5  # Would be great to test effect of num_actions  # TODO
    # A discrete set of sampled continuous actions
    # next_actions = [next_dist.rsample() for _ in range(num_actions)]

    Qs = critic(obs, action)

    # Reduce Q ensemble via min  TODO 'mean minus uncertainty' maybe (https://arxiv.org/pdf/2110.03375.pdf)
    Q = torch.min(*Qs)

    # "Entropy maximization"
    # Entropy - 'aleatory' - uncertainty - randomness in decision-making - keeps exploration active, gradients tractable
    # neg_log_proba = -dist.log_prob(action).mean()  TODO Is this better for entropy?
    entropy = dist.entropy().mean()

    # "Trust region optimization"
    # Policies that change too rapidly per batch are unstable, so we try to bound their temperament a little
    # within a "trust region", ideally one that keeps large gradients from propelling weights beyond their local optima
    # TODO can also try BYOL between actor representations and EMA target representations
    # policy_divergence = torch.kl_div(log_proba, log_proba.detach()).mean()
    # eps = 0.1  # This is the "trust region"
    # TODO Or can try just setting scaling - when >, + when <
    # # Via Lagrangian relaxation TODO wherein metas.trust_region_scale (alpha) is minimized
    # trust_region_bounding = metas.trust_region_scaling.detach() * (eps - policy_divergence)

    # Maximize action-value
    policy_loss = -Q.mean()
    # Maximize entropy
    policy_loss -= entropy_temp * entropy  # TODO metas.entropy_temp (w/ Meta accepting kwargs w/ init vals)
    # Maximize trust region bounding
    # policy_loss -= trust_region_bounding

    if logs is not None:
        assert isinstance(logs, dict)
        logs['policy_loss'] = policy_loss.item()
        logs['avg_action_proba'] = torch.exp(dist.log_prob(action)).mean().item()
        logs['avg_policy_entropy'] = entropy.item()

    return policy_loss
