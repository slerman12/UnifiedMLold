# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from Datasets.Environments.Raw.Wrappers import *


def make(name, frame_stack, action_repeat, max_episode_frames, truncate_episode_frames, seed, train=True):
    domain, task = name.split('_', 1)
    # Overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # Make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # Add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # Add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # Zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
    # Stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    if max_episode_frames and action_repeat:
        max_episode_steps = max_episode_frames // action_repeat
        env = TimeLimit(env, max_episode_len=max_episode_steps)
    if train:
        if truncate_episode_frames and action_repeat:
            truncate_episode_steps = truncate_episode_frames // action_repeat
            env = TimeLimit(env, max_episode_len=truncate_episode_steps, resume=True)
    env = ExtendedTimeStepWrapper(env)
    env = AttributesWrapper(env)
    return env
