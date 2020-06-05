import numpy as np

from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.dataset import episodes_length

from utils import normalize_and_clip


class HER(ReplayMemory):
    def __init__(self, max_size, reward_function, n_additional_goals, sampling):
        self._reward_function = reward_function
        self._future_p = 1. - (1 / (1 + n_additional_goals))

        if sampling == 'final':
            def sample_goals(dataset, i):
                abs_idxs = np.cumsum(episodes_length(dataset)) - 1
                idx = abs_idxs[abs_idxs >= i][0]
                sampled_goals = np.array([dataset[idx][3]['achieved_goal']])

                return sampled_goals
        elif sampling == 'future':
            def sample_goals(dataset, i):
                abs_idxs = np.cumsum(episodes_length(dataset))
                prev_abs_idxs = abs_idxs[abs_idxs <= i]
                episode_end = abs_idxs[len(prev_abs_idxs)]
                idx = np.random.randint(i, episode_end)
                sampled_goals = np.array([dataset[idx][3]['achieved_goal']])
                return sampled_goals
        elif sampling == 'episode':
            def sample_goals(dataset, i):
                abs_idxs = np.cumsum(episodes_length(dataset))
                prev_abs_idxs = abs_idxs[abs_idxs <= i]
                episode_start = prev_abs_idxs[-1] if len(prev_abs_idxs) > 0 else 0
                episode_end = abs_idxs[len(prev_abs_idxs)]
                idx = np.random.randint(episode_start, episode_end)
                sampled_goals = np.array([dataset[idx][3]['achieved_goal']])
                return sampled_goals
        elif sampling == 'random':
            def sample_goals(dataset, _):
                sampled_idxs = np.random.choice(len(dataset))
                sampled_goals = np.array(
                    [dataset[x][3]['achieved_goal'] for x in sampled_idxs]
                )
                return sampled_goals
        else:
            raise ValueError

        self._sample_goals = sample_goals

        self._mu = 0
        self._sigma2 = 0
        self._count = 0

        super().__init__(0, max_size)

    def add(self, dataset):
        idxs_map = dict()
        for i in range(len(dataset)):
            desired_goal = dataset[i][0]['desired_goal']
            state_goal = self._add_transition(dataset, desired_goal, self._idx, i)
            self._update_idx()

            idxs_map[i] = self._idx

            self._update_normalization(state_goal)

        her_idxs = np.where(np.random.rand(len(dataset)) < self._future_p)[0]
        for i in her_idxs:
            sampled_goal = self._sample_goals(dataset, i)
            self._add_transition(dataset, sampled_goal, idxs_map[i], i)

    def get(self, n_samples):
        s, a, r, ss, _, _ = super().get(n_samples)
        s = normalize_and_clip(s, self._mu, self._sigma2)
        ss = normalize_and_clip(ss, self._mu, self._sigma2)

        return s, np.array(a), np.array(r), ss

    def _add_transition(self, dataset, g, replay_idx, idx):
        state_goal = np.append(dataset[idx][0]['observation'], g)
        self._states[replay_idx] = state_goal
        self._actions[replay_idx] = dataset[idx][1]
        next_state_goal = np.append(dataset[idx][3]['observation'], g)
        self._next_states[replay_idx] = next_state_goal

        self._rewards[replay_idx] = self._reward_function(
            dataset[idx][3]['achieved_goal'], g, {}
        )

        return state_goal

    def _update_idx(self):
        self._idx += 1
        if self._idx == self._max_size:
            self._full = True
            self._idx = 0

    def _update_normalization(self, state_goal):
        self._count += 1
        prev_mu = self._mu
        self._mu = (self._mu * (self._count - 1) + state_goal) / self._count
        self._sigma2 = (
            self._sigma2 * (self._count - 1) + (state_goal - prev_mu) * (
                state_goal - self._mu)) / self._count
