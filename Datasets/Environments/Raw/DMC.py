# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import dm_env
from dm_env import StepType, specs
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels

import numpy as np

from collections import deque
from typing import NamedTuple, Any


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def episode_done(self):
        return self.last()

    def get_last(self):
        return self._replace(step_type=StepType.LAST)


class ExtendedAction(NamedTuple):
    shape: Any
    dtype: Any
    minimum: Any
    maximium: Any
    name: Any
    discrete: bool
    num_actions: Any


class ActionWrapper(dm_env.Environment):
    def __init__(self, env, dtype, discrete=False):
        self.env = env
        self.discrete = discrete
        wrapped_action_spec = env.action_spec()
        if discrete:
            num_actions = wrapped_action_spec.shape[-1]
            self._action_spec = ExtendedAction((1,),
                                               dtype,
                                               0,
                                               num_actions,
                                               'action',
                                               True,
                                               num_actions)
        else:
            self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                                   dtype,
                                                   wrapped_action_spec.minimum,
                                                   wrapped_action_spec.maximum,
                                                   'action')

        self.time_step = None

    def step(self, action):
        if hasattr(action, 'astype'):
            action = action.astype(self.env.action_spec().dtype)
        self.time_step = self.env.step(action)
        return self.time_step

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        self.time_step = self.env.reset()
        return self.time_step

    def __getattr__(self, name):
        return getattr(self.env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, action_repeat):
        self.env = env
        self.action_repeat = action_repeat

        self.time_step = None

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self.action_repeat):
            time_step = self.env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        self.time_step = time_step._replace(reward=reward, discount=discount)
        return self.time_step

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self.env.action_spec()

    def reset(self):
        self.time_step = self.env.reset()
        return self.time_step

    def __getattr__(self, name):
        return getattr(self.env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key=None):
        self.env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        self.time_step = None

        wrapped_obs_spec = env.observation_spec()
        if pixels_key is not None:
            assert pixels_key in wrapped_obs_spec
            pixels_shape = wrapped_obs_spec[pixels_key].shape
        else:
            pixels_shape = wrapped_obs_spec.shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation if self._pixels_key is None else time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self.env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        self.time_step = self._transform_observation(time_step)
        return self.time_step

    def step(self, action):
        self.time_step = time_step = self.env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        self.time_step = self._transform_observation(time_step)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self.env.action_spec()

    def close(self):
        self.gym_env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class TruncateWrapper(dm_env.Environment):
    def __init__(self, env, max_episode_steps=np.inf, truncate_episode_steps=np.inf, train=True):
        self.env = env
        self.time_step = None

        self.train = train

        # Truncating/limiting episodes
        self.max_episode_steps = max_episode_steps
        self.truncate_episode_steps = truncate_episode_steps
        self.elapsed_steps = 0
        self.was_not_truncated = False

    def step(self, action):
        self.time_step = time_step = self.env.step(action)
        # Truncate or cut episodes
        self.elapsed_steps += 1
        self.was_not_truncated = time_step.last() or self.elapsed_steps >= self.max_episode_steps
        if self.elapsed_steps >= self.truncate_episode_steps or self.elapsed_steps >= self.max_episode_steps:
            # No truncation for training environments
            if self.train or self.elapsed_steps >= self.max_episode_steps:
                time_step = dm_env.truncation(time_step.reward, time_step.observation, time_step.discount)
        self.time_step = time_step
        return time_step

    def reset(self):
        # Truncate and resume, or reset
        if self.was_not_truncated:
            self.time_step = self.env.reset()
        else:
            self.time_step = self.env.time_step
        self.elapsed_steps = 0
        return self.time_step

    def close(self):
        self.gym_env.close()

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self.env.action_spec()

    def __getattr__(self, name):
        return getattr(self.env, name)


class AugmentAttributesWrapper(dm_env.Environment):
    def __init__(self, env):
        self.env = env

        self.time_step = None

    def step(self, action):
        self.time_step = time_step = self.env.step(action)
        # Augment time_step with extra functionality
        self.time_step = self.augment_time_step(time_step, action)
        return self.time_step

    def reset(self):
        self.time_step = time_step = self.env.reset()
        # Augment time_step with extra functionality
        self.time_step = self.augment_time_step(time_step)
        return self.time_step

    def close(self):
        self.gym_env.close()

    def augment_time_step(self, time_step, action=None):
        if action is None:
            action = np.zeros(self.action_spec['shape'], dtype=self.action_spec['dtype'])
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    @property
    def exp(self):
        return self.time_step.to_attr_dict()

    @property
    def experience(self):
        return self.exp

    def observation_spec(self):
        obs_spec = self.env.observation_spec()
        return self.simplify_spec(obs_spec)

    @property
    def obs_spec(self):
        return self.observation_spec()

    @property
    def obs_shape(self):
        return self.obs_spec['shape']

    @property
    def action_spec(self):
        action_spec = self.env.action_spec()
        return self.simplify_spec(action_spec)

    @property
    def action_shape(self):
        a = self.simplify_spec(self.action_spec)
        return (a['num_actions'],) if self.discrete \
            else self.action_spec['shape']

    def simplify_spec(self, spec):
        # Return spec as a dict of basic primitives (that can be passed into Hydra)
        keys = ['shape', 'dtype', 'name', 'num_actions']
        spec = {key: getattr(spec, key, None) for key in keys}
        spec['dtype'] = spec['dtype'].name
        return spec

    def __getattr__(self, name):
        return getattr(self.env, name)


def make(task, frame_stack, max_episode_frames=None, truncate_episode_frames=None, seed=0, action_repeat=1, train=True):
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
    env = ActionWrapper(env, np.float32)

    # Repeats actions n times
    env = ActionRepeatWrapper(env, action_repeat)

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

    # Truncate-resume or cut epsisodes short
    max_episode_steps = max_episode_frames // action_repeat if max_episode_frames is not None else np.inf
    truncate_episode_steps = truncate_episode_frames // action_repeat if truncate_episode_frames is not None else np.inf
    env = TruncateWrapper(env,
                          max_episode_steps=max_episode_steps,
                          truncate_episode_steps=truncate_episode_steps,
                          train=train)

    # Augment attributes to env and time step, prepare specs for loading by Hydra
    env = AugmentAttributesWrapper(env)

    return env
