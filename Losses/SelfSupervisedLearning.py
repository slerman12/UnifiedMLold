# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F


# TODO can put into TemporalDifferenceLearning (TDLearning) file together with QLearning
def bootstrapLearningBVS(actor, sub_planner, planner, obs, traj_o, plan_discount,
                         traj_a=None, step=None, logs=None):
    target_sub_plan = sub_planner(traj_o)
    with torch.no_grad():
        next_obs = traj_o[:, -1]

        # if traj_a is None:
        #     target_sub_plan = sub_planner.target(traj_o)  # State-based planner  TODO use target?
        # else:
        #     assert step is not None
        #     dist = actor(next_obs, step)
        #     next_action = dist.rsample()
        #     traj_a = torch.cat([traj_a, next_action.unsqueeze(1)], dim=1)
        #     target_sub_plan = sub_planner.target(traj_o, traj_a)  # State-action based planner
        dist = actor(target_sub_plan[:, -1], step)
        next_action = dist.mean
        target_sub_plan[:, -1] = planner.target(target_sub_plan[:, -1], next_action).detach()

    plan_discount = plan_discount ** torch.arange(target_sub_plan.shape[1]).to(obs.device)
    target_plan = torch.einsum('j,ijk->ik', plan_discount, target_sub_plan)
        # target_plan = torch.layer_norm(target_plan, target_plan.shape)

    # if traj_a is None:
    #     sub_plan = sub_planner(obs)  # state-based planner
    # else:
    #     action = traj_a[:, 0]
    #     sub_plan = sub_planner(obs, action)  # state-action based planner
    sub_plan = sub_planner(obs)
    plan = planner(sub_plan, traj_a[:, 0])
    # plan = torch.layer_norm(plan, plan.shape)

    planner_loss = F.mse_loss(plan, target_plan)  # Bellman error

    if logs is not None:
        assert isinstance(logs, dict)
        logs['planner_loss'] = planner_loss.item()

    return planner_loss


def dynamicsLearning(dynamics, projection_g, prediction_q, traj_o, traj_a, logs=None):
    forecasts = dynamics(traj_o, traj_a)  # TODO check dynamic dims for obs and action in CNN
    forecasts = projection_g(forecasts)
    forecasts = prediction_q(forecasts)

    with torch.no_grad():
        projections = projection_g.target(traj_o)  # TODO also encoder target

    dynamics_loss = F.cosine_similarity(forecasts, projections, -1).mean()

    if logs is not None:
        logs['dynamics_loss'] = dynamics_loss

    return dynamics_loss
