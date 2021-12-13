# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import Utils


def ensembleQLearning(actor, critic, obs, action, reward, discount, next_obs, step, ensemble_reduction='min',
                      dist=None, entropy_temp=0,  # 0.03
                      munchausen_temp=0, meta_learn=False, logs=None):  # 0.9
    with torch.no_grad():
        next_dist = actor(next_obs, step)  # Note: not using EMA target for actor

        # If num_actions = 1
        # Would be great to test mean vs sample
        # next_actions = [next_dist.best if critic.discrete else next_dist.mean]

        if critic.discrete:
            # All discrete actions in discrete action space
            next_actions = [torch.full_like(next_dist.best, a) for a in range(actor.action_dim)]
        else:
            num_actions = 5  # Would be great to test effect of num_actions
            # A discrete set of sampled continuous actions
            next_actions = [next_dist.rsample() for _ in range(num_actions)]

        # Ensemble Q learning
        next_Q_ensembles = [critic.target(next_obs, next_action, next_dist)  # Outputs an ensemble per action
                            for next_action in next_actions]

        # How to reduce each ensemble into one Q-value per action
        if ensemble_reduction == 'min':
            next_Q = torch.cat([torch.min(*next_Q_ensemble)
                                for next_Q_ensemble in next_Q_ensembles], -1)
        elif ensemble_reduction == 'mean':
            # See: https://openreview.net/pdf?id=9xhgmsNVHu
            next_Q = torch.cat([(sum(next_Q_ensemble) / len(next_Q_ensemble))
                                for next_Q_ensemble in next_Q_ensembles], -1)
        else:
            # Can also try mean minus uncertainty, where uncertainty is stddev or convolved distances:
            # https://arxiv.org/pdf/2110.03375.pdf
            raise Exception('ensemble reduction', ensemble_reduction, 'not implemented')

        # Value V = expected Q
        next_log_probs = torch.cat([next_dist.log_prob(next_action).mean(-1, keepdim=True)
                                    for next_action in next_actions], -1)
        next_probs = torch.softmax(next_log_probs, -1)
        next_V = (next_Q * next_probs).sum(-1, keepdim=True)

        # TODO consider N-step entropy... +actor(traj_o).entropy(traj_a) ... or should these be added a la Munchausen?
        # "Entropy maximization"
        # Future-action uncertainty maximization in reward
        # Entropy in future decisions means exploring the uncertain, the lesser-explored
        next_entropy = Categorical(next_probs).entropy().mean(-1, keepdim=True)  # Value-based entropy
        # next_entropy = next_dist.entropy().mean(-1, keepdim=True)
        # next_action_log_proba = next_dist.log_prob(next_action).sum(-1, keepdim=True)  # Action-based entropy
        target_Q = reward + (discount * next_V) + (entropy_temp * next_entropy)
        # TODO Q-value itself could be Gaussian, then next_V not needed

        # "Munchausen reward":
        # Current-action certainty maximization in reward, thereby increasing so-called "action-gap" w.r.t. above
        # Furthermore, off-policy sampling of outdated rewards might be mitigated to a degree by on-policy estimate
        # Another salient heuristic: "optimism in the face of uncertainty" (Brafman & Tennenholtz, 2002) literally
        if munchausen_temp != 0:
            if dist is None:
                dist = actor(obs, step)
            # TODO logsumexp trick
            action_log_proba = dist.log_prob(action).mean(-1, keepdim=True)
            # By PRO and trust region insights:
            # action_log_proba = critic(obs, action, dist)
            lo = -1
            target_Q += munchausen_temp * torch.clamp(entropy_temp * action_log_proba, min=lo, max=0)

    context = torch.empty(0)
    if meta_learn:
        _dist = actor(obs)
        context = Utils.one_hot(action, actor.action_dim)
        context = (1 - context) * torch.min(*_dist.Qs) + context * target_Q

    Q_ensemble = critic(obs, action, dist, context.detach())

    # Temporal difference error (via MSE, but could also use Huber)
    td_error = sum([F.mse_loss(Q, target_Q) for Q in Q_ensemble])

    if logs is not None:
        assert isinstance(logs, dict)
        logs['target_q'] = target_Q.mean().item()
        logs.update({f'q{i}': Q.mean().item() for i, Q in enumerate(Q_ensemble)})
        logs['td_error'] = td_error.item()

    return td_error
