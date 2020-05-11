import argparse
import datetime
import pathlib
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from joblib import delayed, Parallel

from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.utils.dataset import compute_J

from core import Core
from fetch_env import FetchEnv
from her import HER


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

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        a = F.tanh(self._a(features3)) * 5.

        return a


def print_epoch(epoch):
    print(
        '################################################################')
    print('Epoch: ', epoch)
    print(
        '----------------------------------------------------------------')


def experiment(exp_id, args):
    np.random.seed(exp_id)

    # MDP
    mdp = FetchEnv(args.name)

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2)

    # Settings
    if args.debug:
        pass
    else:
        initial_replay_size = args.initial_replay_size
        max_replay_size = args.max_replay_size
        batch_size = args.batch_size
        tau = args.tau

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
        replay_memory = HER(args.initial_replay_size, args.max_replay_size)
    else:
        raise ValueError

    # Agent
    if args.alg == 'ddpg':
        agent = DDPG(mdp.info, policy_class, policy_params,
                     actor_params, actor_optimizer, critic_params,
                     args.batch_size, replay_memory, args.tau,
                     args.optimization_steps)
    else:
        raise ValueError

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    print_epoch(0)
    print('--Evaluation--')
    dataset = core.evaluate(n_steps=args.test_epochs, render=False)
    J = compute_J(dataset, mdp.info.gamma)
    print('J: ', np.mean(J))

    for i in range(1, args.max_epochs):
        print_epoch(i)
        print("--Learning--")
        core.learn(n_episodes=args.evaluation_frequency * args.n_cycles,
                   n_episodes_per_fit=args.evaluation_frequency)

        print("--Evaluation--")
        dataset = core.evaluate(n_steps=args.test_epochs, render=False)
        J = compute_J(dataset, mdp.info.gamma)
        print('J: ', np.mean(J))


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Name')
    arg_game.add_argument("--name")
    arg_game.add_argument("--n-exp", type=int)

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=64,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=1000000,
                         help='Max size of the replay memory.')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--alg", type=str, default='her',
                         choices=['ddpg', 'td3', 'sac'])
    arg_alg.add_argument("--replay", type=str, default='her',
                         choices=['her', 'cher', 'dher', 'sphere'])
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
    arg_alg.add_argument("--test-epochs", type=int, default=10,
                         help='Number of epochs for each evaluation.')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--load', type=str,
                           help='Path of the model to be loaded.')
    arg_utils.add_argument('--save', action='store_true',
                           help='Flag specifying whether to save the model.')
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

    out = Parallel(n_jobs=-1)(delayed(experiment)(i, args)
                              for i in range(args.n_exp))

    scores = np.array([o[0] for o in out])

    np.save(folder_name + 'scores.npy', scores)
