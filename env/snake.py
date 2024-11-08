import numpy as np
import gym
from gym.spaces import Discrete

from common.rendering import EnvRender
from env.utils import *


class SnakeDiscreteEnv(gym.Env):
    def __init__(self, size=10, **kwargs):
        print('Snake environment')
        self.size = size
        self.kwargs = kwargs
        self.num_pos = size * size
        print('>>> Size: {}x{}'.format(size, size))
        print('>>> Number of Coins: ', self.kwargs['num_coin'])
        print('>>> Number of Barriers: ', self.kwargs['num_barrier'])
        print('>>> Number of Pools: ', self.kwargs['num_mud'])
        print('>>> Number of Lands: ', self.kwargs['num_land'])
        self.start = 1
        self.pos = 1
        self.last_pos = 1

        self.cv_render = None
        self.ranges = kwargs['rew_setting']
        self.prefix = '_'.join([str(x) for x in kwargs['rew_setting']])
        print('>>> Reward Setting: ', self.prefix)

        self.observation_space = Discrete(self.num_pos)
        self.action_space = Discrete(4)
        print('>>> Observation Space: {}'.format(self.num_pos))
        print('>>> Action Space: {}'.format(4))

    def __initialize(self):
        positions = list(range(self.num_pos))
        np.random.shuffle(positions)

        self.barriers = positions[:self.kwargs['num_barrier']]  # Generate barriers
        positions = positions[self.kwargs['num_barrier']:]

        self.mud = positions[:self.kwargs['num_mud']]           # Generate mud area
        positions = positions[self.kwargs['num_mud']:]

        self.land = positions[:self.kwargs['num_land']]         # Generate land area
        positions = positions[self.kwargs['num_land']:]

        self.coins = positions[:self.kwargs['num_coin']]        # Generate coins
        self.empty = positions[self.kwargs['num_coin']:]

    def reset(self, reuse=False, **kwargs):
        if not reuse:
            self.__initialize()
            np.random.shuffle(self.empty)
            self.pos = self.start = self.empty[0]
        else:
            self.last_pos = self.pos = self.start

        print('>>> Start: ', self.pos, '(king\'s move)')
        print('>>> Coins: ', self.coins)
        print('>>> Lands: ', self.land)
        if self.cv_render is not None:
            self.cv_render.initialize()
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
                x, y, delta_x, delta_y, new_x, new_y),
                end=' ')
        new_pos = coord2state((new_x, new_y), size)
        if not (0 <= new_x < size and 0 <= new_y < size):
            new_pos = pos

        reward, done, terminated = self.get_reward(pos, new_pos)
        if verbose:
            print('\t--> {:>3d} {} {:>3d} {:>+6.2f} {} {}'.format(
                pos, action, new_pos, reward,
                int(done), int(terminated)))
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
