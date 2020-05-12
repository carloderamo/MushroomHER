import numpy as np

from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.dataset import episodes_length


class HER(ReplayMemory):
    def __init__(self, max_size, reward_function, n_additional_goals, sampling):
        self._reward_function = reward_function
        self._n_additional_goals = n_additional_goals

        if sampling == 'final':
            def sample_goals(dataset):
                abs_idxs = np.cumsum(episodes_length(dataset)) - 1
                sampled_idxs = np.random.choice(abs_idxs, size=4)
                sampled_goals = np.array(
                    [dataset[x][0]['achieved_goal'] for x in sampled_idxs]
                )
                return sampled_goals
        else:
            raise ValueError

        self._sample_goals = sample_goals

        super().__init__(0, max_size)

    def add(self, dataset):
        sampled_goals = self._sample_goals(dataset)

        for i in range(len(dataset)):
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
                self._rewards[self._idx] = dataset[i][2]

                self._idx += 1
                if self._idx == self._max_size:
                    self._full = True
                    self._idx = 0
