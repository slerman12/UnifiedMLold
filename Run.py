# Copyright Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import hydra
from hydra.utils import instantiate

import os
from pathlib import Path

import Utils

import torch
# If input sizes consistent, will lead to better performance.
from torch.backends import cudnn
cudnn.benchmark = True


# Hydra conveniently and cleanly manages sys args;
# hyper-param cfg files located in ./Hyperparams

@hydra.main(config_path='Hyperparams', config_name='cfg')
def main(args):
    # Setup

    Utils.set_seed_everywhere(args.seed)

    torch.device(args.device)

    args.root_path, root_path = os.getcwd(), Path.cwd()  # Hydra doesn't support Path types

    # Train, test environments
    env = instantiate(args.environment)  # An instance of DeepMindControl, for example
    test_env = instantiate(args.environment, train=False)

    if (root_path / 'Saved.pt').exists():
        agent, replay = Utils.load(root_path, 'agent', 'replay')
    else:
        for arg in ('obs_shape', 'action_shape', 'discrete', 'obs_spec', 'action_spec'):
            setattr(args, arg, getattr(env, arg))

        # Agent
        agent = instantiate(args.agent)  # An instance of DQNDPGAgent, for example

        # Experience replay
        replay = instantiate(args.replay)  # An instance of PrioritizedExperienceReplay, for example

    # Loggers
    logger = instantiate(args.logger)  # Aggregates per step

    vlogger = instantiate(args.vlogger)

    # Start training
    step = 0
    while step < args.train_steps:
        # Evaluate
        if step % args.evaluate_per_steps == 0:

            for ep in range(args.evaluate_episodes):
                _, logs, vlogs = test_env.rollout(agent.eval(),  # agent.eval() just sets agent.training to False
                                                  vlog=args.log_video)  # todo hydra this?

                logger.log(logs, 'Eval')
            logger.dump_logs('Eval')

            if args.log_video:
                print(vlogs)
                vlogger.vlog(vlogs, 'Eval')

        # Rollout
        experiences, logs, _ = env.rollout(agent.train(), steps=1)  # agent.train() just sets agent.training to True

        replay.add(experiences)

        if env.episode_done:
            logger.log(logs, 'Train', dump=True)

            if env.last_episode_len >= args.nstep:
                replay.add(store=True)  # Only store full episodes

            if args.save_session:
                Utils.save(root_path, agent=agent, replay=replay)

        step = agent.step

        # Update agent
        if step > args.seed_steps and \
                step % args.update_per_steps == 0:

            logs = agent.update(replay)  # Trains the agent

            if args.log_tensorboard:
                logger.log_tensorboard(logs, 'Train')


if __name__ == "__main__":
    main()
