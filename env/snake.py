import numpy as np
import gym
import numpy.random
from gym.spaces import Discrete

from common.rendering import EnvRender
from env.utils import *


class SnakeDiscreteEnv(gym.Env):
    def __init__(self, size=10, num_ladders=15, num_targets=3, num_obstacles=10):
        print('Snake environment')
        self.size = size
        print('>>> Size: {}x{}'.format(size, size))
        self.observation_space = Discrete(size * size)
        self.action_space = Discrete(5)

        self.num_ladders = num_ladders
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles

        self.pos = 1
        self.start = 1
        self.last_pos = 1
        self.targets, self.target_checker = [], []
        self.ladders = {}
        self.obstacles = []

        self.env_info = ''
        self.cv_render = None
        self.reset(verbose=True)

    def __initialize(self, num_states):
        states = list(range(num_states))
        # Generate obstacles
        np.random.shuffle(states)
        obstacles = states[:self.num_obstacles]
        states = states[self.num_obstacles:]
        # Generate ladders
        ladders = {}
        for i in range(self.num_ladders):
            numpy.random.shuffle(states)
            former = states[0]
            num_later = np.random.randint(2, 5)
            later_list = [former, ] + states[1: num_later]
            later_prob = np.random.random(num_later)
            later_prob /= sum(later_prob)
            ladders[former] = [later_list, later_prob]
        print('>>> Ladders: ', self.num_ladders)
        for i, (key, values) in enumerate(ladders.items()):
            [values1, values2] = values
            states.remove(key)
            print(
                '\t{:>3d}: {:<3d} --> '.format(i + 1, key),
                ['{:>3d}: {:>4.3f}'.format(v1, v2) for v1, v2 in zip(values1, values2)]
            )
        # Generate targets
        targets = states[:self.num_targets]
        self.empty = states[self.num_targets:]
        return ladders, targets, obstacles

    def reset(self, reuse=False, verbose=False):
        if not reuse:
            num_states = self.observation_space.n
            self.ladders, self.targets, self.obstacles = self.__initialize(num_states)
            np.random.shuffle(self.empty)
            self.pos = self.start = self.empty[0]
        else:
            self.last_pos = self.pos = self.start
        self.target_checker = self.targets[:]
        if verbose:
            print('>>> Targets: ', ', '.join([str(v) for v in self.targets]))
            print('>>> Obstacles: ', self.num_obstacles)
            print('>>> Start: ', self.pos, '(king\'s move)')
        return self.pos

    def execute_action(self, action, pos, verbose=False):
        """ Up-Down-None-Left-Right move (0-4) """
        motions = [(+0, +1), (-1, +0), (+0, +0), (+1, +0), (+0, -1)]
        size = self.size
        delta_x, delta_y = motions[action]
        [x, y] = state2coord(pos, size)
        new_x, new_y = x + delta_x, y + delta_y
        if verbose:
            print('\t>>> {:>3d} {:>3d} {:>3d} {:>3d} {:>3d} {:>3d}'.format(
                delta_x, delta_y, x, y, new_x, new_y
            ), end=' ')
        new_pos = coord2state((new_x, new_y), size)
        if new_pos in self.obstacles:
            return pos
        if 0 <= new_x < size and 0 <= new_y < size:
            return new_pos
        return pos

    # def execute_action(self, action, pos, verbose=False):
    #     """ King's move (0-8) """
    #     motions = [
    #         (-1, +1), (+0, +1), (+1, +1),
    #         (-1, +0), (+0, +0), (+1, +0),
    #         (-1, -1), (+0, -1), (+1, -1),
    #     ]
    #     size = self.size
    #
    #     delta_x, delta_y = motions[action]
    #     [x, y] = state2coord(pos, size)
    #     new_x, new_y = x+delta_x, y+delta_y
    #     if verbose:
    #         print('\t>>> {:>3d} {:>3d} {:>3d} {:>3d} {:>3d} {:>3d}'.format(
    #             delta_x, delta_y, x, y, new_x, new_y
    #         ), end=' ')
    #     if 0 <= new_x < size and 0 <= new_y < size:
    #         new_pos = coord2state((new_x, new_y), size)
    #         new_pos1 = coord2state((new_x, y), size)
    #         new_pos2 = coord2state((x, new_y), size)
    #         if new_pos in self.obstacles or \
    #             new_pos1 in self.obstacles or \
    #                 new_pos2 in self.obstacles:
    #             return pos
    #         return new_pos
    #     return pos

    def step(self, action, verbose=False):
        old_pos = self.pos
        new_pos = self.execute_action(action, old_pos, verbose=verbose)
        if verbose:
            print('\t>>> {:>3d} {} {:>3d}'.format(old_pos, action, new_pos), end=' ')
        if new_pos in self.ladders.keys():
            random_prob = np.random.random()
            for pos, prob in zip(*self.ladders[new_pos]):
                random_prob -= prob
                if random_prob <= 0.0:
                    new_pos = pos
                    break
        self.last_pos = old_pos
        self.pos = new_pos
        rew, done = self.get_reward(new_pos)
        if verbose:
            print('\t>>> {:>3d} {:>+6.1f} {}'.format(new_pos, rew, int(done)))
        return new_pos, rew, done, {}

    def get_reward1(self, s, scale=1.0):
        if s in self.obstacles:
            return -self.size*2.0-10.0, False
        dists = []
        for target in self.targets:
            dists.append(distance(s, target, size=self.size))
        rew = -float(min(dists))*scale
        return rew, False

    def get_reward(self, s, scale=1.0):
        if s in self.obstacles:
            # return -self.size*2.0-10.0, False
            return 0.0, False
        if s in self.target_checker:
            self.target_checker.remove(s)
            if len(self.target_checker) > 0:
                return 1.0*scale, False
            return 1.0*scale, True
        return 0.0, False

    def render(self, **kwargs):
        if self.cv_render is None:
            self.cv_render = EnvRender(self)
        self.cv_render.draw(**kwargs)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
