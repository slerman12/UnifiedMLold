# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.


def make(task, frame_stack=3, action_repeat=2, max_episode_frames=None, truncate_episode_frames=None,
         train=True, seed=1):
    # Imports in make() to avoid glfw warning when using other envs
    from dm_control import manipulation, suite
    from dm_control.suite.wrappers import action_scale, pixels

    from Datasets.Environments.Raw.Wrappers import ActionSpecWrapper, ActionRepeatWrapper, FrameStackWrapper, \
        TruncateWrapper, AugmentAttributesWrapper

    import numpy as np

    # Load suite and task
    domain, task = task.split('_', 1)
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
        task = f'{domain}_{task}_vision'
        env = manipulation.load(task, seed=seed)
        pixels_key = 'front_close'

    # Add extra info to action specs
    env = ActionSpecWrapper(env, np.float32)

    # Repeats actions n times  (frame skip)
    env = ActionRepeatWrapper(env, action_repeat if train else action_repeat)

    # Rescales actions to range
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

    # Truncate-resume or cut episodes short
    max_episode_steps = max_episode_frames // action_repeat if max_episode_frames else np.inf
    truncate_episode_steps = truncate_episode_frames // action_repeat if truncate_episode_frames else np.inf
    env = TruncateWrapper(env,
                          max_episode_steps=max_episode_steps,
                          truncate_episode_steps=truncate_episode_steps,
                          train=train)

    # Augment attributes to env and time step, prepare specs for loading by Hydra
    env = AugmentAttributesWrapper(env)

    return env
