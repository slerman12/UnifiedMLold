# Copyright (c) Sam Lerman. All Rights Reserved.
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

    args.root_path, root_path = os.getcwd(), Path.cwd()  # Hydra doesn't support Path types

    # All agents can convert seamlessly between RL or classification

    # RL vs. classification is automatically inferred based on task,
    # e.g., task=dmc/humanoid_walk (RL), task=classify/mnist (classification)

    if args.RL:
        # Reinforcement Learning
        reinforce(args, root_path)
    else:
        # Classification
        classify(args, root_path)


def reinforce(args, root_path):
    # Train, test environments
    env = instantiate(args.environment)  # An instance of DeepMindControl, for example
    generalize = instantiate(args.environment, train=False)

    # Load
    if (root_path / 'Saved.pt').exists():
        agent, replay = Utils.load(root_path, 'agent', 'replay')

        agent = Utils.to_agent(agent).to(args.device)
    else:
        for arg in ('obs_shape', 'action_shape', 'discrete', 'obs_spec', 'action_spec'):
            setattr(args, arg, getattr(env, arg))

        # Agent
        agent = instantiate(args.agent).to(args.device)  # An instance of DQNDPGAgent, for example

        # Experience replay
        replay = instantiate(args.replay)  # An instance of PrioritizedExperienceReplay, for example

    # Loggers
    logger = instantiate(args.logger)

    vlogger = instantiate(args.vlogger)

    # Start training
    step = 0
    while step < args.train_steps:
        # Evaluate
        if step % args.evaluate_per_steps == 0:

            for ep in range(args.evaluate_episodes):
                _, logs, vlogs = generalize.rollout(agent.eval(),  # agent.eval() just sets agent.training to False
                                                    vlog=args.log_video)

                logger.log(logs, 'Eval')
            logger.dump_logs('Eval')

            if args.log_video:
                vlogger.dump_vlogs(vlogs, f'{agent.step}.mp4')

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


def classify(args, root_path):
    # Agent
    agent = Utils.load(root_path,
                       'agent') if (root_path / 'Saved.pt').exists() \
        else instantiate(args.agent)  # An instance of DQNDPGAgent, for example

    # Convert to classifier
    agent = Utils.to_classifier(agent)

    # Experience replay (train, test)
    replay, generalize = instantiate(args.replay)  # An instance of Cifar_10, for example

    # Loggers
    logger = instantiate(args.logger)

    # Start training
    step = 0
    while step < args.train_steps:

        if step % args.evaluate_per_steps == 0:

            # Evaluate
            logs = agent.eval().update(generalize)

            logger.log(logs, 'Eval', dump=True)

            # Save
            if args.save_session:
                Utils.save(root_path, agent=agent)

        # Train
        logs = agent.train().update(replay)

        logger.log(logs, 'Train', dump=True)

        if args.log_tensorboard:
            logger.log_tensorboard(logs, 'Train')

        step += 1

    # Death


if __name__ == "__main__":
    main()
