# Copyright Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from collections import deque, namedtuple
from typing import Any, NamedTuple, Optional
import gym
from gym import spaces
import dm_env
import numpy as np
from dm_env import StepType, specs


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

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedAction(NamedTuple):
    shape: Any
    dtype: Any
    minimum: Any
    maximium: Any
    name: Any
    discrete: bool
    num_actions: Any

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, action_repeat):
        self._env = env
        self.action_repeat = action_repeat

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self.action_repeat):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key=None):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

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
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def close(self):
        self.gym_env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype, discrete=False):
        self._env = env
        self.discrete = discrete
        wrapped_action_spec = env.action_spec()
        if not discrete:
            self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                                   dtype,
                                                   wrapped_action_spec.minimum,
                                                   wrapped_action_spec.maximum,
                                                   'action')
        else:
            num_actions = wrapped_action_spec.shape[-1]
            self._action_spec = ExtendedAction((1,),
                                               dtype,
                                               0,
                                               num_actions,
                                               'action',
                                               True,
                                               num_actions)

    def step(self, action):
        if hasattr(action, 'astype'):
            action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class AttributesWrapper(dm_env.Environment):
    def __init__(self, env):
        self.env = env

    @property
    def exp(self):
        return self.env.time_step

    @property
    def obs_spec(self):
        return self.observation_spec()

    @property
    def obs_shape(self):
        return self.observation_spec().shape

    @property
    def action_shape(self):
        return (self.action_spec.num_actions,) \
            if self.discrete else self.action_spec.shape

    @property
    def experience(self):
        return self.exp

    def observation_spec(self):
        obs_spec = self.env.observation_spec()
        keys = ['shape', 'dtype', 'name']
        Spec = namedtuple('Spec', ' '.join(keys))
        return Spec(*[getattr(obs_spec, key, None) for key in keys])

    @property
    def action_spec(self):
        action_spec = self.env.action_spec()
        keys = ['shape', 'dtype', 'minimum', 'maximum', 'name', 'discrete', 'num_actions']
        Spec = namedtuple('Spec', ' '.join(keys))
        return Spec(*[getattr(action_spec, key, None) for key in keys])

    def step(self, action):
        self.env.step()

    def reset(self):
        self.env.reset()

    def __getattr__(self, name):
        return getattr(self.env, name)


class TimeLimit(dm_env.Environment):
    """Return done before end of episode. If resume is True,
    resume episode without reset if episode not actually done
    """
    def __init__(self, env, max_episode_len=None, resume=False):
        self._env = env
        self._max_episode_len = max_episode_len
        self._elapsed_steps = 0
        self.was_not_truncated = True
        self.resume = resume

    def step(self, action):
        time_step = self._env.step(action)
        self.was_not_truncated = not self.resume or time_step.last()
        self._elapsed_steps += 1
        if self._max_episode_len:
            if self._elapsed_steps >= self._max_episode_len:
                # time_step = time_step.get_last()   # todo make sure same thing as below edit: unless extended t step
                time_step = dm_env.truncation(time_step.reward, time_step.observation, time_step.discount)
        self.time_step = time_step
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        self._elapsed_steps = 0
        if self.was_not_truncated:
            time_step = self._env.reset()
            self.time_step = time_step
        # else:
        # no-op step to advance from terminal/lost life state  todo also don't need to turn off this stuff for eval?
        # time_step = self._env.step(0)  # todo shouldn't take step, just return the current time_step
        return self.time_step

    def close(self):
        self.gym_env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self.time_step = None

    def reset(self):
        self.time_step = time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def close(self):
        self.gym_env.close()

    def step(self, action):
        self.time_step = time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
