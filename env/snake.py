import numpy as np
import gym
from gym.spaces import Discrete

from common.rendering import EnvRender
from env.utils import *


def n_binary(x, n):
    binary_x = bin(x)[2:]
    while len(binary_x) < n:
        binary_x = '0' + binary_x
    return binary_x


class SnakeDiscreteEnv(gym.Env):
    def __init__(self, size=10, **kwargs):
        print('Snake environment')
        self.size = size
        self.kwargs = kwargs
        self.num_pos = size * size
        print('>>> Size: {}x{}'.format(size, size))
        self.start = 1
        self.pos = 1
        self.last_pos = 1
        self.coin_checker = {}

        self.cv_render = None
        self.ranges = None
        self.reset(verbose=True)

        self.__build_table()
        self.observation_space = Discrete(len(self.state_dict))
        self.action_space = Discrete(4)
        print('>>> Observation Space: {}'.format(len(self.state_dict)))
        print('>>> Action Space: {}'.format(4))

    def __initialize(self):
        kwargs = self.kwargs
        poses = list(range(self.num_pos))
        np.random.shuffle(poses)

        self.barriers = poses[:kwargs['num_barrier']]  # Generate barriers
        poses = poses[kwargs['num_barrier']:]
        self.mud = poses[:kwargs['num_mud']]           # Generate mud area
        poses = poses[kwargs['num_mud']:]
        ladders = {}                                   # Generate ladders
        for i in range(kwargs['num_ladder']):
            np.random.shuffle(poses)
            former = poses[0]
            num_later = np.random.randint(2, 5)
            later_list = [former, ] + poses[1: num_later]
            later_prob = np.random.random(num_later)
            later_prob /= sum(later_prob)
            ladders[former] = [later_list, later_prob]
        print('>>> Ladders: ', kwargs['num_ladder'])
        for i, (key, values) in enumerate(ladders.items()):
            [values1, values2] = values
            poses.remove(key)
            print(
                '\t{:>3d}: {:<3d} --> '.format(i + 1, key),
                ['{:>3d}: {:>4.3f}'.format(v1, v2) for v1, v2 in zip(values1, values2)]
            )
        self.ladders = ladders
        self.coins = poses[:kwargs['num_coin']]  # Generate coins
        self.empty = poses[kwargs['num_coin']:]

    def __build_table(self):
        num_pos, num_coin = self.num_pos, self.kwargs['num_coin']
        num_layer = int(''.join(['1' for _ in range(num_coin)]), 2) + 1  # 8
        state_dict, state_list = {}, []
        for pos in range(num_pos):
            idx = self.coins.index(pos) if pos in self.coins else None
            for i in range(num_layer):
                state = pos + i * num_pos
                status = n_binary(i, num_coin)
                if i >= num_layer - 1 and idx is None:
                    continue
                if idx is not None and status[idx] != '1':
                    continue
                state_dict[state] = {'pos': pos, 'status': status}
                state_list.append(state)
        self.state_dict = state_dict
        self.state_list = state_list

    def get_state(self, pos=None, coin_checker=None):
        if pos is None:
            pos = self.pos
        if coin_checker is None:
            coin_checker = self.coin_checker

        status = ''.join([str(status) for status in coin_checker.values()])
        num_layer = int(status, 2)
        return pos + num_layer * self.num_pos

    def reset(self, reuse=False, verbose=False):
        if not reuse:
            self.__initialize()
            np.random.shuffle(self.empty)
            self.pos = self.start = self.empty[0]
            self.ranges = (-1.0, 0.0, +1.0)  # s_b, s_m, s_g
        else:
            self.last_pos = self.pos = self.start
        self.coin_checker = {pos: 0 for pos in self.coins}
        if verbose:
            print('>>> Coins: ', ', '.join(['{}({})'.format(k, v) for k, v in self.coin_checker.items()]))
            print('>>> Barriers: ', self.kwargs['num_barrier'])
            print('>>> Start: ', self.pos, '(king\'s move)')
        return self.get_state()

    def step(self, action, verbose=False):
        """ Up-Down-None-Left-Right move (0-4) """
        pos = self.pos
        size = self.size
        # motions = [(+0, +1), (-1, +0), (+0, +0), (+1, +0), (+0, -1)]
        motions = [(+0, +1), (-1, +0), (+1, +0), (+0, -1)]

        delta_x, delta_y = motions[action]
        [x, y] = state2coord(pos, size)
        new_x, new_y = x + delta_x, y + delta_y
        if verbose:
            print('\t>>> ({:>3d},{:>3d}) ({:>3d},{:>3d}) ({:>3d},{:>3d})'.format(
                x, y, delta_x, delta_y, new_x, new_y
            ), end=' ')
        new_pos = coord2state((new_x, new_y), size)
        if not (0 <= new_x < size and 0 <= new_y < size):
            new_pos = pos
        if new_pos in self.ladders.keys():
            random_prob = np.random.random()
            for pos, prob in zip(*self.ladders[new_pos]):
                random_prob -= prob
                if random_prob <= 0.0:
                    new_pos = pos
                    break
        if new_pos in self.coin_checker.keys():
            if self.coin_checker[new_pos] == 0:
                self.coin_checker[new_pos] = 1
        reward, done, terminated = self.get_reward(pos=new_pos)
        if verbose:
            print('\t--> {:>3d} {} {:>3d} {:>+6.2f} {} {}'.format(
                pos, action, new_pos, reward, int(done), int(terminated))
            )
        self.last_pos = pos
        self.pos = new_pos
        return self.get_state(), reward, done, terminated

    def get_reward(self, state=None, pos=None):
        if pos is None:
            pos = self.state_dict[state]['pos']

        if pos in self.barriers:
            reward, done, terminated = -1.0 * self.ranges[0], False, True
        elif pos in self.mud:
            reward, done, terminated = -1.0, False, False
        elif pos in self.coin_checker.keys():
            done = all([status == 1 for status in self.coin_checker.values()])
            terminated = False
            reward = +1.0*self.ranges[2] if done else -1.0 * self.ranges[1]
        else:
            reward, done, terminated = -1.0 * self.ranges[1], False, False
        return reward, done, terminated

    def render(self, **kwargs):
        if self.cv_render is None:
            self.cv_render = EnvRender(self)
        self.cv_render.draw(**kwargs)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
