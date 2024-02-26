import numpy as np
import gym
import numpy.random
from gym.spaces import Discrete

from .visual import CVRender
from .utils import distance


class SnakeEnv(gym.Env):
    def __init__(self, size=10, num_ladders=15, num_targets=3):
        print('Snake environment')
        self.size = size
        print('>>> Size: {}x{}'.format(size, size))
        self.observation_space = Discrete(size * size)
        self.action_space = Discrete(5)

        self.num_ladders = num_ladders
        self.num_targets = num_targets

        self.pos = 1
        self.start = 1
        self.records = []
        self.targets, self.target_checker = [], []
        self.ladders = {}
        self.x = 10.0
        self.y = -1.0

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

    def reset(self, reuse=False, verbose=False, **kwargs):
        if not reuse:
            num_states = self.observation_space.n
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
        return self.pos

    def execute_action(self, action, pos):
        # 0-north, 1-south, 2-east, 3-west, 4-no move
        # todo: king's moves (southeast, northwest, southwest, northeast)
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
        return new_pos, rew, done, {}

    def get_reward(self, s):
        if s in self.target_checker:
            self.target_checker.remove(s)
            if len(self.target_checker) > 0:
                return self.x, False
            return self.x, True
        return self.y, False

    def get_reward2(self, s):
        # if s in self.target_checker:
        #     self.target_checker.remove(s)
        #
        # if len(self.target_checker) <= 0:
        #     return 0.0, True

        if s in self.target_checker:
            return 0.0, True

        rew = 0.0
        dists = []
        for target in self.target_checker:
            dist = distance(s, target, self.size)
            dists.append(dist)
        rew -= min(dists) * 0.01
        return rew, False

    def render(self, **kwargs):
        if self.cv_render is None:
            self.cv_render = CVRender(self)
        self.cv_render.draw(last_pos=self.records, **kwargs)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
