# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import random
import datetime
import io
import traceback
from collections import namedtuple

import numpy as np

import torch
from torch.utils.data import IterableDataset


class ExperienceReplay:
    def __init__(self, storage_dir, obs_spec, action_spec, capacity, batch_size, num_workers,
                 save, nstep, discount):

        self.storage = ExperienceStorage(storage_dir, obs_spec, action_spec,
                                         capacity=capacity // max(1, num_workers),
                                         num_workers=num_workers,
                                         fetch_every=1000,
                                         save=save)

        self._replay = None
        self.storage.sample = self.sample
        self.storage.process = self.process

        def worker_init_fn(worker_id):
            seed = np.random.get_state()[1][0] + worker_id
            np.random.seed(seed)
            random.seed(seed)

        self.loading = torch.utils.data.DataLoader(dataset=self.storage,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   pin_memory=True,
                                                   worker_init_fn=worker_init_fn)

        self.nstep = nstep
        self.discount = discount

    def add(self, experiences=None, store=False):
        self.storage.add(experiences, store=store)

    @property
    def replay(self):
        if self._replay is None:
            self._replay = iter(self.loading)
        return self._replay

    def sample(self, episode_names, metrics=None):
        episode_name = random.choice(episode_names)  # Uniform sampling of experiences
        return episode_name

    def process(self, episode):
        episode_len = next(iter(episode.values())).shape[0] - 1
        idx = np.random.randint(0, episode_len - self.nstep + 1) + 1

        # Transition
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self.nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])

        # Trajectory
        traj_o = episode["observation"][idx - 1:idx + self.nstep]
        traj_a = episode["action"][idx:idx + self.nstep]
        traj_r = episode["reward"][idx:idx + self.nstep]

        # Compute cumulative discounted reward
        for i in range(self.nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self.discount

        return obs, action, reward, discount, next_obs, traj_o, traj_a, traj_r

    def __iter__(self):
        return self.replay.__iter__()

    def __next__(self):
        return self.replay.__next__()


# Stores episodes (to files in system)
# Multi-cpu workers iteratively and efficiently build batches of experience in parallel (from files)
class ExperienceStorage(IterableDataset):
    def __init__(self, storage_dir, obs_spec, action_spec, capacity, num_workers, fetch_every, save=False):
        # Episode storage

        self.storage_dir = storage_dir
        storage_dir.mkdir(exist_ok=True)

        self.num_episodes = 0
        self.num_experiences_total = 0

        Spec = namedtuple("Spec", "shape dtype name")
        self.specs = (obs_spec, action_spec,
                      Spec((1,), np.float32, 'reward'),
                      Spec((1,), np.float32, 'discount'))

        self.episode = {spec.name: [] for spec in self.specs}
        self.episode_len = 0

        # Dataset construction via parallel workers

        self.episode_names = []
        self.episodes = dict()

        self.num_experiences_stored = 0
        self.capacity = capacity

        self.num_workers = max(1, num_workers)

        self.fetch_every = fetch_every
        self.samples_since_last_fetch = fetch_every

        self.save = save

    def __len__(self):
        return self.num_experiences_total

    def add(self, experiences=None, store=False):
        if experiences is None:
            experiences = []

        # An "episode" of experiences
        assert isinstance(experiences, (list, tuple))

        for exp in experiences:
            for spec in self.specs:
                if np.isscalar(exp[spec.name]):
                    exp[spec.name] = np.full(spec.shape, exp[spec.name], spec.dtype)
                self.episode[spec.name].append(exp[spec.name])
                assert spec.shape == exp[spec.name].shape
                assert spec.dtype == exp[spec.name].dtype

        self.episode_len += len(experiences)

        if store:
            self.store_episode()

    def store_episode(self):
        for spec in self.specs:
            self.episode[spec.name] = np.array(self.episode[spec.name], spec.dtype)

        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        episode_name = f'{timestamp}_{self.num_episodes}_{self.episode_len}.npz'

        # Save episode
        save_path = self.storage_dir / episode_name
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **self.episode)
            buffer.seek(0)
            with save_path.open('wb') as f:
                f.write(buffer.read())

        self.num_episodes += 1
        self.num_experiences_total += self.episode_len
        self.episode = {spec.name: [] for spec in self.specs}
        self.episode_len = 0

    def load_episode(self, episode_name):
        try:
            with episode_name.open('rb') as episode_file:
                episode = np.load(episode_file)
                episode = {key: episode[key] for key in episode.keys()}
        except Exception:
            return False

        episode_len = next(iter(episode.values())).shape[0] - 1

        while episode_len + self.num_experiences_stored > self.capacity:
            early_episode_name = self.episode_names.pop(0)
            early_episode = self.episodes.pop(early_episode_name)
            self.num_experiences_stored -= episode_len(early_episode)
            # deletes early episode file
            early_episode_name.unlink(missing_ok=True)

        self.episode_names.append(episode_name)
        # TODO Book-keep corresponding metrics for prioritized sampling
        self.episode_names.sort()
        self.episodes[episode_name] = episode
        self.num_experiences_stored += episode_len

        if not self.save:
            episode_name.unlink(missing_ok=True)  # deletes file

        return True

    # Populates workers with up-to-date data
    def worker_fetch_episodes(self):
        if self.samples_since_last_fetch < self.fetch_every:
            return

        self.samples_since_last_fetch = 0

        try:
            worker_id = torch.utils.data.get_worker_info().id
        except Exception:
            worker_id = 0

        episode_names = sorted(self.storage_dir.glob('*.npz'), reverse=True)
        num_fetched = 0
        for episode_name in episode_names:
            episode_idx, episode_len = [int(x) for x in episode_name.stem.split('_')[1:]]
            if episode_idx % self.num_workers != worker_id:  # Each worker stores their own dedicated data
                continue
            if episode_name in self.episodes.keys():  # Don't store redundantly
                break
            if num_fetched + episode_len > self.capacity:  # Don't overfill
                break
            num_fetched += episode_len
            if not self.load_episode(episode_name):
                break  # Resolve conflicts

    def sample(self, episode_names, metrics=None):  # Can be over-ridden from ExperienceReplay
        episode_name = random.choice(episode_names)  # Uniform sampling of experiences
        return episode_name

    def process(self, episode):  # Can be over-ridden from ExperienceReplay
        experience = tuple(episode[spec.name] for spec in self.specs)
        return experience

    def fetch_sample_process(self):
        try:
            self.worker_fetch_episodes()  # Populate workers with up-to-date data
        except Exception:
            traceback.print_exc()

        self.samples_since_last_fetch += 1

        episode_name = self.sample(self.episode_names)  # Sample an episode

        episode = self.episodes[episode_name]

        return self.process(episode)  # Process episode into an experience

    def __iter__(self):
        while True:
            yield self.fetch_sample_process()
