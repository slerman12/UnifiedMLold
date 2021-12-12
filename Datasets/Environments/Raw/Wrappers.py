import dm_env
from dm_env import StepType, specs

import numpy as np

from collections import deque
from typing import NamedTuple, Any


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    # step: int = None

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


class ActionSpecWrapper(dm_env.Environment):
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

    def step(self, action):
        if hasattr(action, 'astype'):
            action = action.astype(self.env.action_spec().dtype)
        time_step = self.env.step(action)
        return time_step

    def reset(self):
        time_step = self.env.reset()
        return time_step

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self.env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, action_repeat):
        self.env = env
        self.action_repeat = action_repeat

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self.action_repeat):
            time_step = self.env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        time_step = time_step._replace(reward=reward, discount=discount)
        return time_step

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self.env.action_spec()

    def reset(self):
        time_step = self.env.reset()
        return time_step

    def __getattr__(self, name):
        return getattr(self.env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key=None):
        self.env = env
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
        time_step = self.env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        time_step = self._transform_observation(time_step)
        return time_step

    def step(self, action):
        time_step = self.env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self.env.action_spec()

    def close(self):
        self.gym_env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


# Note: Could technically do in Run.py just by setting rollout steps to truncate_episode_steps and always add to replay
class TruncateWrapper(dm_env.Environment):
    def __init__(self, env, max_episode_steps=np.inf, truncate_episode_steps=np.inf, train=True):
        self.env = env

        self.train = train

        # Truncating/limiting episodes
        self.max_episode_steps = max_episode_steps
        self.truncate_episode_steps = truncate_episode_steps
        self.elapsed_steps = 0
        self.was_not_truncated = True

    def step(self, action):
        time_step = self.env.step(action)
        # Truncate or cut episodes
        self.elapsed_steps += 1
        self.was_not_truncated = time_step.last() or self.elapsed_steps >= self.max_episode_steps
        if self.elapsed_steps >= self.truncate_episode_steps or self.elapsed_steps >= self.max_episode_steps:
            # No truncation for eval environments
            if self.train or self.elapsed_steps >= self.max_episode_steps:  # Let max episode cutoff apply to eval
                time_step = dm_env.truncation(time_step.reward, time_step.observation, time_step.discount)
        self.time_step = time_step
        return time_step

    def reset(self):
        # Truncate and resume, or reset
        if self.was_not_truncated:
            self.time_step = self.env.reset()
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


# Access a dict with attribute or key
class AttrDict(dict):
    def __init__(self, d):
        super(AttrDict, self).__init__()
        self.__dict__ = self
        self.update(d)


class AugmentAttributesWrapper(dm_env.Environment):
    def __init__(self, env):
        self.env = env

        self.time_step = None

    def step(self, action):
        time_step = self.env.step(action)
        # Augment time_step with extra functionality
        self.time_step = self.augment_time_step(time_step, action)
        return self.to_attr_dict(self.time_step)

    def reset(self):
        time_step = self.env.reset()
        # Augment time_step (experience) with extra functionality
        self.time_step = self.augment_time_step(time_step)
        return self.to_attr_dict(self.time_step)

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
        return self.to_attr_dict(self.time_step)

    def to_attr_dict(self, exp):
        keys = ['step_type', 'reward', 'discount', 'observation', 'action',
                'first', 'mid', 'last', 'episode_done', 'get_last']
        return AttrDict({key: getattr(exp, key, None) for key in keys})

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
        a = self.action_spec
        return (a['num_actions'],) if self.discrete else a['shape']

    def simplify_spec(self, spec):
        # Return spec as a dict of basic primitives (that can be passed into Hydra)
        keys = ['shape', 'dtype', 'name', 'num_actions']
        spec = {key: getattr(spec, key, None) for key in keys}
        if not isinstance(spec['dtype'], str):
            spec['dtype'] = spec['dtype'].name
        return spec

    def __getattr__(self, name):
        return getattr(self.env, name)