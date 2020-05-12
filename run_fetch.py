import argparse
import datetime
import pathlib
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from joblib import delayed, Parallel

from mushroom_rl.utils.dataset import compute_J, episodes_length
from mushroom_rl.core import Core

from ddpg import DDPG
from fetch_env import FetchEnv
from her import HER
from policy import EpsilonGaussianPolicy


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features = 64

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._q = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._q.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):


        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        q = self._q(features3)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features = 64

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._a = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._a.weight,
                                gain=nn.init.calculate_gain('tanh'))

    def forward(self, state, output_features=False):


        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        a = torch.tanh(self._a(features3)) * 5.

        if output_features:
            return features3, a
        else:
            return a


def get_stats(dataset, gamma):
    J = compute_J(dataset, gamma)
    abs_idxs = np.cumsum(episodes_length(dataset)) - 1
    S = list()
    for idx in abs_idxs:
        S.append(dataset[idx][3]['info']['is_success'])

    print('J: ', np.mean(J))
    print('S: ', np.mean(S))

    return J, S


def print_epoch(epoch):
    print(
        '################################################################')
    print('Epoch: ', epoch)
    print(
        '----------------------------------------------------------------')


def experiment(exp_id, folder_name, args):
    np.random.seed()

    # MDP
    mdp = FetchEnv(args.name)
    n_actions = mdp.info.action_space.shape[0]
    action_range = mdp.info.action_space.high - mdp.info.action_space.low

    # Policy
    policy_class = EpsilonGaussianPolicy
    sigma_policy = np.eye(n_actions) * 1e-6
    policy_params = dict(sigma=sigma_policy, epsilon=.2,
                         action_space=mdp.info.action_space)

    # Settings
    if args.debug:
        evaluation_frequency = 16
        n_cycles = 4
        max_epochs = 32
        test_episodes = 4
        max_replay_size = 1000
        batch_size = 4
    else:
        evaluation_frequency = args.evaluation_frequency
        n_cycles = args.n_cycles
        max_epochs = args.max_epochs
        test_episodes = args.test_episodes
        max_replay_size = args.max_replay_size
        batch_size = args.batch_size

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=ActorNetwork,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape,
                        use_cuda=args.use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': .001}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': .001}},
                         loss=F.mse_loss,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=args.use_cuda)

    if args.replay == 'her':
        replay_memory = HER(max_replay_size, mdp.compute_reward,
                            args.n_additional_goals, args.sampling)
    else:
        raise ValueError

    # Agent
    if args.alg == 'ddpg':
        agent = DDPG(mdp.info, policy_class, policy_params,
                     actor_params, actor_optimizer, critic_params,
                     batch_size, replay_memory, args.tau,
                     args.optimization_steps)
    else:
        raise ValueError

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    scores = list()
    successes = list()

    print_epoch(0)
    print('--Evaluation--')
    agent.policy.set_weights(agent._target_actor_approximator.get_weights())
    dataset = core.evaluate(n_episodes=args.test_episodes, render=args.render,
                            quiet=args.quiet)
    j, s = get_stats(dataset, mdp.info.gamma)
    scores += j
    successes += s

    np.save(folder_name + '/scores_%d.npy' % exp_id, scores)
    np.save(folder_name + '/successes_%d.npy' % exp_id, successes)

    for i in range(1, max_epochs):
        print_epoch(i)
        print("--Learning--")
        agent.policy.set_weights(agent._actor_approximator.get_weights())
        sigma_policy = np.diag(action_range * .05)
        agent.policy.set_sigma(sigma_policy)
        core.learn(n_episodes=evaluation_frequency * n_cycles,
                   n_episodes_per_fit=evaluation_frequency, quiet=args.quiet)

        print("--Evaluation--")
        agent.policy.set_weights(agent._target_actor_approximator.get_weights())
        sigma_policy = np.eye(n_actions) * 1e-6
        agent.policy.set_sigma(sigma_policy)
        dataset = core.evaluate(n_episodes=test_episodes, render=args.render,
                                quiet=args.quiet)
        j, s = get_stats(dataset, mdp.info.gamma)
        scores += j
        successes += s

        np.save(folder_name + '/scores.npy', scores)
        np.save(folder_name + '/successes.npy', successes)

    return scores, successes


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Name')
    arg_game.add_argument("--name", type=str, default='FetchReach-v1')
    arg_game.add_argument("--n-exp", type=int, default=1)

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--max-replay-size", type=int, default=1000000,
                         help='Max size of the replay memory.')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--alg", type=str, default='ddpg',
                         choices=['ddpg', 'td3', 'sac'])
    arg_alg.add_argument("--replay", type=str, default='her',
                         choices=['her', 'cher', 'dher', 'sphere'])
    arg_alg.add_argument("--n-additional-goals", type=int, default=4)
    arg_alg.add_argument("--sampling", type=str, default='final',
                         choices=['final', 'future', 'episode', 'random'])
    arg_alg.add_argument("--batch-size", type=int, default=128,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--tau", type=float, default=.95)
    arg_alg.add_argument("--n-cycles", type=int, default=50)
    arg_alg.add_argument("--evaluation-frequency", type=int, default=16,
                         help='Number of learning episodes before each evaluation.'
                              'This number represents an epoch.')
    arg_alg.add_argument("--optimization-steps", type=int, default=40)
    arg_alg.add_argument("--max-epochs", type=int, default=200,
                         help='Total number of learning steps.')
    arg_alg.add_argument("--test-episodes", type=int, default=10,
                         help='Number of epochs for each evaluation.')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--render', action='store_true',
                           help='Flag specifying whether to render the game.')
    arg_utils.add_argument('--quiet', action='store_true',
                           help='Flag specifying whether to hide the progress'
                                'bar.')
    arg_utils.add_argument('--debug', action='store_true',
                           help='Flag specifying whether the script has to be'
                                'run in debug mode.')

    args = parser.parse_args()

    folder_name = './logs/' + args.alg + '_' + args.name + '/' + \
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'
    pathlib.Path(folder_name).mkdir(parents=True)

    with open(folder_name + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)

    out = Parallel(n_jobs=-1)(delayed(experiment)(i, folder_name, args)
                              for i in range(args.n_exp))

    scores = np.array([o[0] for o in out])
    success = np.array([o[1] for o in out])

    np.save(folder_name + 'scores.npy', scores)
    np.save(folder_name + 'success.npy', success)
