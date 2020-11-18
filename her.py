import numpy as np
from mushroom_rl.core import Serializable

from utils import normalize_and_clip


class HER(Serializable):
    def __init__(self, horizon, observation_space, goal_space, action_space,
                 max_size, reward_function, n_additional_goals):
        assert max_size % horizon == 0

        self._initial_size = 0
        self._horizon = horizon
        self._max_size = max_size
        self._n_episodes = self._max_size // self._horizon

        self.reset()

        self._add_save_attr(
            _initial_size='pickle',
            _horizon='pickle',
            _max_size='pickle',
            _n_episodes='pickle',
            _idx_episode='pickle!',
            _full='pickle!',
            _states='pickle!',
            _actions='pickle!',
            _rewards='pickle!',
            _next_states='pickle!',
            _absorbing='pickle!',
            _last='pickle!'
        )

        self._reward_function = reward_function
        self._future_p = 1. - (1 / (1 + n_additional_goals))

        self._mu = 0
        self._sigma2 = 0
        self._count = 0

    def add(self, dataset):
        for i in range(len(dataset)):
            for j in range(self._horizon):
                self._states[self._idx_episode][j] = dataset[i][0]
                self._actions[self._idx_episode][j] = dataset[i][1]
                self._rewards[self._idx_episode][j] = 0
                self._next_states[self._idx_episode][j] = dataset[i][3]

            self._update_idx_episode()

    def get(self, n_samples):
        s = list()
        a = list()
        r = list()
        ss = list()
        s_augmented = list()
        ss_augmented = list()

        idx_episodes = np.random.randint(self.size, size=n_samples)
        idx_samples = np.random.randint(self._horizon, size=n_samples)
        idx_sample_goals = np.random.randint(idx_samples, self._horizon, size=n_samples)
        her_idxs_episode = np.where(np.random.rand(n_samples) < self._future_p)[0]
        for i in range(n_samples):
            idx_ep = idx_episodes[i]
            idx_sam = idx_samples[i]
            s.append(self._states[idx_ep][idx_sam])
            a.append(self._actions[idx_ep][idx_sam])
            ss.append(self._next_states[idx_ep][idx_sam])

            idx_sam_goal = idx_sample_goals[i]
            if i in her_idxs_episode:
                s_goal = self._states[idx_ep][idx_sam_goal]['achieved_goal']
                ss_goal = self._next_states[idx_ep][idx_sam_goal]['achieved_goal']
            else:
                s_goal, ss_goal = s[i]['desired_goal'], ss[i]['desired_goal']

            s_augmented.append(np.append(s[i]['observation'], s_goal))
            ss_augmented.append(np.append(ss[i]['observation'], ss_goal))

            self._update_normalization(s_augmented[i])
            self._update_normalization(ss_augmented[i])

            r.append(self._reward_function(ss[i]['achieved_goal'], ss_goal, {}))

        s_augmented = normalize_and_clip(np.array(s_augmented), self._mu, self._sigma2)
        ss_augmented = normalize_and_clip(np.array(ss_augmented), self._mu, self._sigma2)

        return s_augmented, np.array(a), np.array(r), ss_augmented

    def reset(self):
        self._idx_episode = 0
        self._full = False
        self._states = [[None for _ in range(self._horizon)] for _ in range(self._n_episodes)]
        self._actions = [[None for _ in range(self._horizon)] for _ in range(self._n_episodes)]
        self._rewards = [[None for _ in range(self._horizon)] for _ in range(self._n_episodes)]
        self._next_states = [[None for _ in range(self._horizon)] for _ in range(self._n_episodes)]

    def _update_idx_episode(self):
        self._idx_episode += 1
        if self._idx_episode == self._n_episodes:
            self._full = True
            self._idx_episode = 0

    def _update_normalization(self, state_goal):
        self._count += 1
        prev_mu = self._mu
        self._mu = (self._mu * (self._count - 1) + state_goal) / self._count
        self._sigma2 = (
            self._sigma2 * (self._count - 1) + (state_goal - prev_mu) * (
                state_goal - self._mu)) / self._count

    @property
    def initialized(self):
        return self.size > self._initial_size

    @property
    def size(self):
        return self._idx_episode if not self._full else self._n_episodes

    def _post_load(self):
        if self._full is None:
            self.reset()
