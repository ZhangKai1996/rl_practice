import numpy as np
from contextlib import contextmanager
import time

from .agent import ModelFreeAgent


@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print('{} COST:{}'.format(name, end - start))


class MonteCarlo(object):
    def __init__(self, env, epsilon=0.0, max_len=-1):
        self.epsilon = epsilon
        self.agent = ModelFreeAgent(env)
        self.max_len = max_len
        self.env = env

    def update(self):
        for _ in range(10):
            for _ in range(100):
                self.__evaluation()
            self.__improvement()
        return self.agent.pi

    def __evaluation(self):
        agent = self.agent
        env = self.env

        state = env.reset(reuse=True)
        episode = []
        while True:
            ac = agent.play(state, self.epsilon)
            next_state, reward, done, _ = env.step(ac)
            episode.append((state, ac, reward))
            state = next_state
            if done or len(episode) >= self.max_len:
                break

        value = []
        return_val = 0
        for (s, ac, rew) in reversed(episode):
            return_val = return_val * agent.gamma + rew
            value.append((s, ac, return_val))

        # every visit
        for (s, ac, ret) in reversed(value):
            agent.value_n[s, ac] += 1
            agent.value_q[s, ac] += (ret - agent.value_q[s, ac]) / agent.value_n[s, ac]

    def __improvement(self):
        agent = self.agent

        new_policy = np.zeros_like(agent.pi)
        for s in range(agent.s_len):
            idx = np.argmax(agent.value_q[s, :])
            new_policy[s, idx] = 1

        if np.all(np.equal(new_policy, agent.pi)):
            return False
        agent.pi = new_policy
        return True
