import numpy as np

from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.dataset import episodes_length


class HER(ReplayMemory):
    def __init__(self, max_size, reward_function, n_additional_goals, sampling):
        self._reward_function = reward_function
        self._n_additional_goals = n_additional_goals

        if sampling == 'final':
            def sample_goals(dataset, _):
                abs_idxs = np.cumsum(episodes_length(dataset)) - 1
                sampled_idxs = np.random.choice(abs_idxs,
                                                size=self._n_additional_goals)
                sampled_goals = np.array(
                    [dataset[x][3]['achieved_goal'] for x in sampled_idxs]
                )
                return sampled_goals
        elif sampling == 'future':
            def sample_goals(dataset, i):
                abs_idxs = np.cumsum(episodes_length(dataset))
                prev_abs_idxs = abs_idxs[abs_idxs <= i]
                episode_end = abs_idxs[len(prev_abs_idxs)]
                print(i, episode_end)
                sampled_idxs = np.random.randint(i, episode_end,
                                                 size=self._n_additional_goals)
                sampled_goals = np.array(
                    [dataset[x][3]['achieved_goal'] for x in sampled_idxs]
                )
                return sampled_goals
        elif sampling == 'episode':
            def sample_goals(dataset, i):
                abs_idxs = np.cumsum(episodes_length(dataset))
                prev_abs_idxs = abs_idxs[abs_idxs <= i]
                episode_start = prev_abs_idxs[-1] if len(prev_abs_idxs) > 0 else 0
                episode_end = abs_idxs[len(prev_abs_idxs)]
                sampled_idxs = np.random.randint(episode_start, episode_end,
                                                 size=self._n_additional_goals)
                sampled_goals = np.array(
                    [dataset[x][3]['achieved_goal'] for x in sampled_idxs]
                )
                return sampled_goals
        elif sampling == 'random':
            def sample_goals(dataset, _):
                sampled_idxs = np.random.choice(len(dataset),
                                                size=self._n_additional_goals)
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
        for i in range(len(dataset)):
            sampled_goals = self._sample_goals(dataset, i)
            goals = np.append([dataset[i][0]['desired_goal']], sampled_goals, axis=0)
            for g in goals:
                state_goal = np.append(dataset[i][0]['observation'], g)
                self._states[self._idx] = state_goal
                self._actions[self._idx] = dataset[i][1]
                next_state_goal = np.append(dataset[i][3]['observation'], g)
                self._next_states[self._idx] = next_state_goal
                self._absorbing[self._idx] = dataset[i][4]
                self._last[self._idx] = dataset[i][5]

                self._rewards[self._idx] = self._reward_function(
                    dataset[i][3]['achieved_goal'], g, {}
                )

                self._idx += 1
                if self._idx == self._max_size:
                    self._full = True
                    self._idx = 0

                self._count += 1
                prev_mu = self._mu
                self._mu = (
                    self._mu * (self._count - 1) + state_goal
                ) / self._count
                self._sigma2 = (self._sigma2 * (self._count - 1) + (
                    state_goal - prev_mu) * (state_goal - self._mu)
                ) / self._count

    def get(self, n_samples):
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        for i in np.random.randint(self.size, size=n_samples):
            s.append(np.array(self._states[i]))
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(np.array(self._next_states[i]))
            ab.append(self._absorbing[i])
            last.append(self._last[i])

        s = self.normalize_and_clip(s)
        ss = self.normalize_and_clip(ss)

        return s, np.array(a), np.array(r), ss, np.array(ab), np.array(last)

    def normalize_and_clip(self, s):
        return np.clip(
            (np.array(s) - self._mu) / (np.sqrt(self._sigma2) + 1e-10), -5., 5.
        )
