import numpy as np
import gym
import numpy.random
from gym.spaces import Discrete, Box

from common import EnvRender
from .utils import state2coord, distance


class SnakeEnvV1(gym.Env):
    def __init__(self, size=10, num_ladders=15, num_targets=3):
        print('Snake environment')
        self.size = size
        self.num_ladders = num_ladders
        self.num_targets = num_targets
        print('>>> Size: {}x{}'.format(size, size))

        self.observation_space = Box(-1.0, 1.0, shape=(2*(num_targets+1), ))
        self.action_space = Discrete(5)

        self.pos = 1
        self.start = 1
        self.records = []
        self.targets, self.target_checker = [], []
        self.ladders = {}

        self.env_info = ''
        self.cv_render = None
        self.reset(verbose=True)

    def __initialize(self, num_states):
        states = list(range(num_states))

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
            print(
                '\t{:>3d}: {:<3d} --> '.format(i + 1, key),
                ['{:>3d}: {:>4.3f}'.format(v1, v2) for v1, v2 in zip(values1, values2)]
            )
        # Generate targets
        targets = []
        while True:
            np.random.shuffle(states)
            state = states[0]
            if state not in ladders.keys() and state not in targets:
                targets.append(state)
            if len(targets) >= self.num_targets:
                break
        return ladders, targets

    def get_obs(self):
        coord1 = state2coord(self.pos, self.size, reg=True)
        state = [coord1]
        for target in self.targets:
            if target not in self.target_checker:
                state.append([-1.0, -1.0])
            else:
                coord2 = state2coord(target, self.size, reg=True)
                state.append([x1-x2 for x1, x2 in zip(coord1, coord2)])
        return np.concatenate(state)

    def reset(self, reuse=False, verbose=False, **kwargs):
        if not reuse:
            num_states = self.size * self.size
            self.ladders, self.targets = self.__initialize(num_states)
            while True:
                self.pos = self.start = np.random.randint(num_states)
                if self.pos not in self.ladders.keys():
                    break
        else:
            self.pos = self.start
            self.records = []
        self.target_checker = self.targets[:]
        if verbose:
            print('>>> Targets: ', '-->'.join([str(v) for v in self.targets]))
            print('>>> Start: ', self.pos, '(0-up, 1-down, 2-right, 3-left, 4-no move)')
        self.records.append((self.pos, ''))
        return self.get_obs()

    def execute_action(self, action, pos):
        # 0-north, 1-south, 2-east, 3-west, 4-no move
        if action == 0:
            if pos + 1 > self.size: pos -= self.size
        elif action == 1:
            if pos + 1 <= self.size * (self.size - 1): pos += self.size
        elif action == 2:
            if (pos + 1) % self.size != 0: pos += 1
        elif action == 3:
            if pos % self.size != 0: pos -= 1
        elif action == 4:
            pos = pos
        else:
            raise NotImplementedError
        return pos

    def step(self, action, verbose=False):
        old_pos = self.pos
        new_pos = self.execute_action(action, old_pos)
        if verbose:
            print('{:>3d} {} {:>3d}'.format(old_pos, action, new_pos), end=' ')
        if new_pos in self.ladders.keys():
            random_prob = np.random.random()
            for pos, prob in zip(*self.ladders[new_pos]):
                random_prob -= prob
                if random_prob <= 0.0:
                    new_pos = pos
                    break
        self.pos = new_pos
        self.records.append((new_pos, ''))
        rew, done = self.get_reward(new_pos)
        if verbose:
            print('{:>3d} {:>+6.1f} {}'.format(new_pos, rew, int(done)))
        return self.get_obs(), rew, done, {}

    def get_reward(self, s):
        if s in self.target_checker:
            self.target_checker.remove(s)
            if len(self.target_checker) > 0:
                return 1.0, False
            return 1.0, True
        return -1.0, False

    # def get_reward(self, s):
    #     dists = []
    #     for target in self.target_checker:
    #         dists.append(distance(s, target, self.size))
    #     rew = -min(dists)*0.1
    #     if s in self.target_checker:
    #         self.target_checker.remove(s)
    #     return rew, len(self.target_checker) <= 0

    def render(self, **kwargs):
        if self.cv_render is None:
            self.cv_render = EnvRender(self)
        self.cv_render.draw(last_pos=self.records, **kwargs)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
