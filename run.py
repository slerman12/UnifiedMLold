# Copyright Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import hydra
from hydra.utils import instantiate

from pathlib import Path

from Logger import Logger
# from Vlogger import VideoRecorder  # M1 Mac: comment out freeimage imports in imageio/plugins/_init_
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

    root_path = Path.cwd()

    # Train, test environments
    env = instantiate(args.environment)  # An instance of DeepMindControl, for example
    test_env = instantiate(args.environment, train=False)

    if (root_path / 'Saved.pt').exists():
        agent, replay = Utils.load(root_path, 'agent', 'replay')
    else:
        # Agent
        for arg in ('obs_shape', 'action_shape', 'discrete'):
            setattr(args.agent, arg, getattr(env, arg))

        agent = instantiate(args.agent)  # An instance of DQNDPGAgent, for example

        # Experience replay
        replay = instantiate(args.replay,  # An instance of PrioritizedExperienceReplay, for example
                             storage_dir=root_path / 'buffer',
                             obs_spec=env.obs_spec,
                             action_spec= env.action_spec)

    # Loggers
    logger = Logger(root_path, use_tensorboard=args.log_tensorboard)  # Aggregates per step

    # vlogger = VideoRecorder(root_path if args.log_video else None)

    # Start training
    step = 0
    while step < args.train_steps:
        # Rollout
        experiences, logs, _ = env.rollout(agent.train(), steps=1)  # agent.train() just sets agent.training to True

        replay.add(experiences, store=False)

        logger.log(logs, 'Train')

        if env.episode_done:
            # logger.dump_logs()

            if env.last_episode_len >= args.nstep:
                replay.add(store=True)  # Only store full episodes

        if args.save_session:
            Utils.save(root_path, agent=agent, replay=replay)

        # Update agent
        if step > args.seed_steps:
            logs = agent.update(replay)  # Trains the agent

            if args.log_tensorboard:
                logger.log(logs, 'Train')

        step = agent.step

        # Evaluate
        if step % args.evaluate_per_steps == 0:
            for ep in range(args.evaluate_episodes):
                _, logs, vlogs = test_env.rollout(agent.eval())   # agent.eval() just sets agent.training to False

                logger.log(logs, 'Eval')

                # if args.log_video:
                #     vlogger.vlog(vlogs)


if __name__ == "__main__":
    main()
