from copy import copy

import gym

from mushroom_rl.environments.gym_env import Environment, MDPInfo
from mushroom_rl.utils.spaces import *


class FetchEnv(Environment):
    def __init__(self, name, reward_type):
        self._close_at_stop = True

        self.env = gym.make(name)
        self.env.env.reward_type = reward_type

        action_space = self._convert_gym_action_space(self.env.action_space)
        observation_space, self.goal_space = self._convert_gym_observation_space(
            self.env.observation_space
        )
        gamma = .98
        horizon = self.env._max_episode_steps
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        if isinstance(action_space, Discrete):
            self._convert_action = lambda a: a[0]
        else:
            self._convert_action = lambda a: a

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            state = self.env.reset()

            return self.clip_dictionary_state(state)
        else:
            self.env.reset()
            self.env.state = state

            return self.clip_dictionary_state(state)

    def step(self, action):
        action = self._convert_action(action)

        state, reward, _, info = self.env.step(action)
        state = self.clip_dictionary_state(state)
        state['info'] = copy(info)

        return state, reward, False, info

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def stop(self):
        try:
            if self._close_at_stop:
                self.env.close()
        except:
            pass

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    @staticmethod
    def clip_dictionary_state(state, value=200):
        for k, v in state.items():
            state[k] = np.clip(v, -value, value)

        return state

    @staticmethod
    def _convert_gym_observation_space(space):
        return Box(low=space['observation'].low, high=space['observation'].high, shape=space['observation'].shape),\
               Box(low=space['desired_goal'].low, high=space['desired_goal'].high, shape=space['desired_goal'].shape)

    @staticmethod
    def _convert_gym_action_space(space):
        return Box(low=space.low, high=space.high, shape=space.shape)
