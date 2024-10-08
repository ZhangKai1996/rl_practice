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

        self.cv_render = None
        self.ranges = None
        self.reset(verbose=True)

        self.observation_space = Discrete(self.num_pos)
        self.action_space = Discrete(4)
        print('>>> Observation Space: {}'.format(self.num_pos))
        print('>>> Action Space: {}'.format(4))

    def __initialize(self):
        kwargs = self.kwargs
        poses = list(range(self.num_pos))
        np.random.shuffle(poses)

        # self.barriers = [2, 7, 17, 22]
        # self.land = [1, 6, 5, 10, 15, 20, 21, 16,
        #              12,
        #              8, 3, 4, 9, 14, 19, 18, 23]
        # self.mud = [11, 13, ]
        # self.coins = [24, ]
        # self.empty = [0, ]
        # self.ladders = {}

        self.barriers = poses[:kwargs['num_barrier']]  # Generate barriers
        poses = poses[kwargs['num_barrier']:]
        self.mud = poses[:kwargs['num_mud']]           # Generate mud area
        poses = poses[kwargs['num_mud']:]
        self.land = poses[:kwargs['num_land']]         # Generate land area
        poses = poses[kwargs['num_land']:]
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

    def reset(self, reuse=False, verbose=False):
        if not reuse:
            self.__initialize()
            # np.random.shuffle(self.empty)
            self.pos = self.start = self.empty[0]
            self.ranges = (-10.0, -1.0, 0.0, +1.0, +10.0)
        else:
            self.last_pos = self.pos = self.start
        if verbose:
            print('>>> Coins: ', self.coins)
            print('>>> Barriers: ', self.kwargs['num_barrier'])
            print('>>> Start: ', self.pos, '(king\'s move)')
        return self.pos

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
        reward, done, terminated = self.get_reward(pos, new_pos)
        if verbose:
            print('\t--> {:>3d} {} {:>3d} {:>+6.2f} {} {}'.format(
                pos, action, new_pos, reward, int(done), int(terminated))
            )
        self.last_pos = pos
        self.pos = new_pos
        return self.pos, reward, done, terminated

    def get_reward(self, pos, new_pos):
        if new_pos in self.barriers:
            return self.ranges[0], False, True
        if new_pos in self.mud:
            return self.ranges[1], False, False

        if new_pos in self.coins:
            if pos in self.coins:
                return self.ranges[2], True, False
            if pos in self.land and pos != self.land[-1]:
                return self.ranges[3] * 0.9, True, False
            return self.ranges[4], True, False

        if new_pos in self.land:
            if pos in self.coins:
                return self.ranges[2], True, False
            if pos in self.land:
                diff = self.land.index(new_pos) - self.land.index(pos)
                if diff <= 0:
                    return self.ranges[2], False, False
                if diff == 1:
                    return self.ranges[3], False, False
                return self.ranges[3] * 0.9, False, False
            return self.ranges[3], False, False
        return self.ranges[2], False, False

    def render(self, **kwargs):
        if self.cv_render is None:
            self.cv_render = EnvRender(self)
        self.cv_render.draw(**kwargs)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
