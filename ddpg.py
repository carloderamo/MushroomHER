import numpy as np

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator

from copy import deepcopy


class DDPG(DeepAC):
    def __init__(self, mdp_info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params, batch_size,
                 replay_memory, tau, optimization_steps, policy_delay=1,
                 critic_fit_params=None):
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = batch_size
        self._tau = tau
        self._optimization_steps = optimization_steps
        self._policy_delay = policy_delay
        self._fit_count = 0

        self._replay_memory = replay_memory

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(TorchApproximator,
                                             **actor_params)
        self._target_actor_approximator = Regressor(TorchApproximator,
                                                    **target_actor_params)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)
        self._init_target(self._actor_approximator,
                          self._target_actor_approximator)

        policy = policy_class(self._actor_approximator, **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='numpy',
            _tau='numpy',
            _policy_delay='numpy',
            _fit_count='numpy',
            _replay_memory='pickle',
            _critic_approximator='pickle',
            _target_critic_approximator='pickle',
            _actor_approximator='pickle',
            _target_actor_approximator='pickle'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset):
        self._replay_memory.add(dataset)

        for _ in range(self._optimization_steps):
            state, action, reward, next_state, absorbing, _ =\
                self._replay_memory.get(self._batch_size)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next
            q = np.clip(q, -1 / (1 - self.mdp_info.gamma), 0)

            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            if self._fit_count % self._policy_delay == 0:
                loss = self._loss(state)
                self._optimize_actor_parameters(loss)

            self._fit_count += 1

        self._update_target(self._critic_approximator,
                            self._target_critic_approximator)
        self._update_target(self._actor_approximator,
                            self._target_actor_approximator)

    def _loss(self, state):
        action = self._actor_approximator(state, output_tensor=True,
                                          unscaled=True)
        q = self._critic_approximator(state, action, output_tensor=True)

        return -q.mean() + action.norm() ** 2

    def _next_q(self, next_state, absorbing):
        a = self._target_actor_approximator(next_state)

        q = self._target_critic_approximator.predict(next_state, a)
        q *= 1 - absorbing

        return q

    def _post_load(self):
        if self._optimizer is not None:
            self._parameters = list(self._actor_approximator.model.network.parameters())

    def draw_action(self, state):
        state = np.append(state['observation'], state['desired_goal'])

        return self.policy.draw_action(state)
