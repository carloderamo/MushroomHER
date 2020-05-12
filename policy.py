import numpy as np

from mushroom_rl.policy.gaussian_policy import GaussianPolicy


class EpsilonGaussianPolicy(GaussianPolicy):
    def __init__(self, mu, sigma, epsilon, action_space):
        self._epsilon = epsilon
        self._action_space = action_space

        super().__init__(mu, sigma)

    def draw_action(self, state):
        if np.random.rand() < self._epsilon:
            return np.random.uniform(low=self._action_space.low,
                                     high=self._action_space.high)
        else:
            mu, sigma = self._compute_multivariate_gaussian(state)[:2]

            return np.random.multivariate_normal(mu, sigma)