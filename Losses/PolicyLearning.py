# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch.distributions import Categorical
from torch.nn import functional as F


def deepPolicyGradient(actor, critic, obs, step, entropy_temp=0.2, trust_region_temp=0.2, dist=None, logs=None):
    if dist is None:
        dist = actor(obs, step)

    # action = dist.mean
    # Would be great to test effect of mean vs sample
    # action = dist.rsample()  # Traditional way is to sample, but only necessary if learnable entropy/stddev, right?

    # Qs = critic(obs, action)

    num_actions = 5  # Would be great to test effect of num_actions  # TODO
    # A discrete set of sampled continuous actions
    actions = [dist.rsample() for _ in range(num_actions)]

    # Ensemble Q learning
    Q_ensembles = [critic.target(obs, action, dist)  # Outputs an ensemble per action
                   for action in actions]

    # How to reduce each ensemble into one Q-value per action
    ensemble_reduction = 'min'
    if ensemble_reduction == 'min':
        Q = torch.cat([torch.min(*Q_ensemble)
                       for Q_ensemble in Q_ensembles], -1)
    elif ensemble_reduction == 'mean':
        # See: https://openreview.net/pdf?id=9xhgmsNVHu
        Q = torch.cat([(sum(Q_ensemble) / len(Q_ensemble))
                       for Q_ensemble in Q_ensembles], -1)
    else:
        # Can also try mean minus uncertainty, where uncertainty is stddev or convolved distances:
        # https://arxiv.org/pdf/2110.03375.pdf
        raise Exception('ensemble reduction', ensemble_reduction, 'not implemented')

    # Value V = expected Q
    log_probs = torch.cat([dist.log_prob(action).mean(-1, keepdim=True)
                           for action in actions], -1)
    probs = torch.softmax(log_probs, -1)
    V = (Q * probs).sum(-1, keepdim=True)

    # Reduce Q ensemble via min  TODO 'mean minus uncertainty' maybe (https://arxiv.org/pdf/2110.03375.pdf)
    # Q = torch.min(*Qs)

    # "Entropy maximization"
    # Entropy - 'aleatory' - uncertainty - randomness in decision-making - keeps exploration active, gradients tractable
    # neg_log_proba = -dist.log_prob(action).mean()  TODO Is this better for entropy?
    # entropy = dist.entropy().mean()  # TODO metas.entropy_temp (w/ Meta accepting kwargs w/ init vals)
    # Alternatively, via scatter sampling:
    entropy = Categorical(probs).entropy().mean()

    # "Trust region optimization"
    # Policies that change too rapidly per batch are unstable, so we try to bound their temperament a little
    # within a "trust region", ideally one that keeps large gradients from propelling weights beyond their local optima
    # TODO can also try BYOL between actor representations and EMA target representations
    # log_proba = dist.log_prob(action).mean()
    log_proba = sum([dist.log_prob(action).mean() for action in actions]) / len(actions)
    policy_divergence = torch.kl_div(log_proba, log_proba.detach()).mean()
    # By PRO:
    # policy_divergence = F.mse_loss(V, V.detach()).mean()
    eps = 0.1  # This is the "trust region"
    # TODO Or can try just setting scaling - when >, + when <
    # trust_region_meta = (policy_divergence > eps) * 2 - 1
    trust_region_meta = policy_divergence - eps  # Can also try
    # # Via Lagrangian relaxation TODO wherein metas.trust_region_scale (alpha) is minimized
    # trust_region_meta = metas.trust_region_meta
    trust_region_bounding = trust_region_meta.detach() * (eps - policy_divergence)
    # if kld > eps, meta > 0, maximize by decreasing kld  ✓
    # if kld < eps, meta > 0, maximize by decreasing kld
    # if kld > eps, meta < 0, maximize by increasing kld
    # if kld < eps, meta < 0, maximize by increasing kld  ✓

    # Maximize action-value
    # policy_loss = -Q.mean()
    # Maximize expected value
    policy_loss = -V.mean()
    # Maximize entropy
    policy_loss -= entropy_temp * entropy
    # Maximize trust region bounding
    policy_loss -= trust_region_temp * trust_region_bounding

    if logs is not None:
        assert isinstance(logs, dict)
        logs['policy_loss'] = policy_loss.item()
        # logs['avg_action_proba'] = torch.exp(dist.log_prob(action)).mean().item()
        logs['avg_action_proba'] = torch.exp(log_proba).mean().item()
        logs['avg_policy_entropy'] = entropy.item()

    return policy_loss
