import csv

from tqdm import tqdm
import numpy as np

from algo.agent import NetworkAgent


class PolicyGradient(object):
    def __init__(self, env, gamma=0.99, epsilon=0.0, alpha=1e-2, max_len=100,
                 improve_iter=-1, **kwargs):
        self.env = env
        self.name = 'VPG'
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_len = max_len
        self.agent = NetworkAgent(env)
        self.i_iter = int(1e6) if improve_iter <= 0 else improve_iter

    def update(self):
        for i in tqdm(range(self.i_iter), desc='VPG update:'):
            state = self.env.reset(reuse=True)

            experiences = []
            while True:
                action = self.agent.play(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                experiences.append((state, action, reward, done))
                state = next_state
                if done or len(experiences) >= self.max_len:
                    break

            self.agent.learn(experiences)
        return self.agent.pi
