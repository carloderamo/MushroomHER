import numpy as np

from mushroom_rl.policy.gaussian_policy import GaussianPolicy


class EpsilonGaussianPolicy(GaussianPolicy):
    def __init__(self, mu, sigma, epsilon, max_action):
        self._epsilon = epsilon
        self._max_action = max_action

        super().__init__(mu, sigma)

    def draw_action(self, state):
        if np.random.rand() < self._epsilon:
            return np.random.uniform(low=-self._max_action,
                                     high=self._max_action)
        else:
            mu, sigma = self._compute_multivariate_gaussian(state)[:2]

            return np.clip(np.random.multivariate_normal(mu, sigma),
                           -self._max_action, self._max_action)

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon
