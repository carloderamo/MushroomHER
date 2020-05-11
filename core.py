import numpy as np
from mpi4py import MPI

from mushroom_rl.core import Core as CoreMSH


class Core(CoreMSH):
    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None,
              n_episodes_per_fit=None, render=False, quiet=False):
        pass

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=False):
        pass

    def _step(self, render):
        self._state = np.append(self._state, self._goal)

        super()._step(render)

    def reset(self, initial_states=None):
        self._goal = pass

        super().reset(initial_states)
