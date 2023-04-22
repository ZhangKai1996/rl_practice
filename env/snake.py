import numpy as np
import gym
from gym.spaces import Discrete

from .utils import CVRender


class SnakeEnv(gym.Env):
    """
    1. Single agent and single target
    2. Barriers and Traps
    """
    def __init__(self, ladder_num=0, size=10, trap_num=10):
        print('Snake environment')
        self.size = size
        self.observation_space = Discrete(size*size)
        self.action_space = Discrete(4)

        pos_list = list(range(0, size*size))  # state = {0,1,2,...,size*size-1]
        np.random.shuffle(pos_list)
        self.ladders = {}
        for i in range(ladder_num):
            former = pos_list[2 * i]
            later = pos_list[2 * i + 1]
            self.ladders[former] = later
            self.ladders[later] = former
            print('ladder{}: {}-{}'.format(len(self.ladders) // 2, former, later))

        self.pos = 1
        self.start = 1
        np.random.shuffle(pos_list)
        self.target = pos_list.pop(0)
        self.traps = pos_list[:trap_num]

        self.last_points = []
        self.env_info = 'target:{},ladders:{},size:{}x{}'.format(
            self.target, ladder_num, size, size)
        self.cv_render = None

    def reset(self, reuse=False):
        if not reuse:
            while True:
                self.pos = self.start = np.random.randint(self.observation_space.n)
                if self.pos in self.traps:
                    continue

                if self.pos not in self.ladders.keys():
                    break
            self.env_info = 'ENV start:{},'.format(self.pos)+self.env_info
            print(self.env_info)
        else:
            self.pos = self.start
        return self.pos

    def execute_action(self, action, pos):
        size = self.size

        # 0-north, 1-south, 2-east, 3-west
        # todo: king's moves (southeast, northwest, southwest, northeast)
        if action == 0:
            if pos + 1 > size: pos -= size
        elif action == 1:
            if pos + 1 <= size * (size - 1): pos += size
        elif action == 2:
            if (pos + 1) % size != 0: pos += 1
        elif action == 3:
            if pos % size != 0: pos -= 1
        else:
            raise NotImplementedError
        return pos

    def step(self, action, has_text=True):
        old_pos = self.pos
        new_pos = self.execute_action(action, old_pos)

        in_trap = new_pos in self.traps
        in_target = new_pos == self.target
        in_ladder = new_pos in self.ladders.keys()
        if not in_trap and not in_target and in_ladder:
            new_pos = self.ladders[new_pos]
        self.pos = new_pos

        reward, done = self.reward(new_pos)
        text = '>>> state{:>3d},action:{},next state:{:>3d},reward:{:>+4d},done:{}'.format(
            old_pos, action, new_pos, reward, int(done))
        if has_text:
            print(text)
        self.last_points.append((old_pos, text))
        if len(self.last_points) >= 10:
            self.last_points.pop(0)
        return new_pos, reward, done, {}

    def reward(self, s):
        if s == self.target:
            return 100, True

        if s in self.traps:
            return -100, True

        return -1, False

    def render(self, mode='human'):
        if self.cv_render is None:
            self.cv_render = CVRender(self)

        # self.cv_render.draw(last_pos=self.last_points)
        self.cv_render.draw(show=True, last_pos=self.last_points)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
