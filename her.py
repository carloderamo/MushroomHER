import numpy as np

from mushroom_rl.utils.replay_memory import ReplayMemory


class HER(ReplayMemory):
    def __init__(self, max_size, reward_function, sampling):
        self._reward_function = reward_function
        self._sampling = sampling

        super().__init__(0, max_size)

    def add(self, dataset):
        for i in range(len(dataset)):
            state_goal = np.append(dataset[i][0]['observation'],
                                   dataset[i][0]['desired_goal'])
            self._states[self._idx] = state_goal
            self._actions[self._idx] = dataset[i][1]
            next_state_goal = np.append(dataset[i][3]['observation'],
                                        dataset[i][3]['desired_goal'])
            self._next_states[self._idx] = next_state_goal
            self._absorbing[self._idx] = dataset[i][4]
            self._last[self._idx] = dataset[i][5]

            self._rewards[self._idx] = self._reward_function(
                dataset[i][3]['achieved_goal'],
                dataset[i][3]['desired_goal'],
                {}
            )
            self._rewards[self._idx] = dataset[i][2]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

        if self._sampling == 'final':
            pass
