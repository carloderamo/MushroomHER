import gym

from mushroom_rl.environments.gym_env import Environment, MDPInfo
from mushroom_rl.utils.spaces import *


class FetchEnv(Environment):
    def __init__(self, name, reward_type):
        self._close_at_stop = True

        self.env = gym.make(name)
        self.env._max_episode_steps = np.inf
        self.env.env.reward_type = reward_type

        action_space = self._convert_gym_action_space(self.env.action_space)
        observation_space = self._convert_gym_observation_space(
            self.env.observation_space
        )
        gamma = .98
        horizon = 50
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        if isinstance(action_space, Discrete):
            self._convert_action = lambda a: a[0]
        else:
            self._convert_action = lambda a: a

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            return self.env.reset()
        else:
            self.env.reset()
            self.env.state = state

            return np.clip(state, -200, 200)

    def step(self, action):
        action = self._convert_action(action)

        state, reward, absorbing, info = self.env.step(action)
        state['info'] = info

        return np.clip(state, -200, 200), reward, absorbing, info

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
    def _convert_gym_observation_space(space):
        low = np.append(space['observation'].low, space['desired_goal'].low)
        high = np.append(space['observation'].high, space['desired_goal'].high)
        shape = (space['observation'].shape[0] + space['desired_goal'].shape[0],)

        return Box(low=low, high=high, shape=shape)

    @staticmethod
    def _convert_gym_action_space(space):
        return Box(low=space.low, high=space.high, shape=space.shape)
