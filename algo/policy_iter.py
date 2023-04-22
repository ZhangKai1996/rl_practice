import time

import numpy as np

from .agent import TableAgent


class PolicyIteration(object):
    def __init__(self, env, eval_max_iter=-1):
        self.agent = TableAgent(env)
        self.eval_max_iter = eval_max_iter if eval_max_iter > 0 else int(1e4)
        self.gamma = 0.8

    def update(self):
        start = time.time()
        while True:
            self.__evaluation()
            if not self.__improvement():
                break
        print('Time consumption:', time.time()-start)
        return self.agent.pi

    def __evaluation(self):
        agent = self.agent
        gamma = self.gamma
        max_iter = self.eval_max_iter

        count = 0
        while count <= max_iter:
            old_value_pi = agent.value_pi
            new_value_pi = old_value_pi.copy()
            r = agent.r
            # v(s) for each state s
            for s in range(agent.s_len):
                p_act = agent.pi[s]
                new_value = 0.0
                for act in range(agent.a_len):
                    prob = p_act[act]
                    new_value += prob * np.dot(agent.p[act, s, :], r + gamma * old_value_pi)
                new_value_pi[s] = new_value
            diff = np.sqrt(np.sum(np.power(old_value_pi - new_value_pi, 2)))
            count += 1
            if diff < 1e-6:
                break
            agent.value_pi = new_value_pi

    def __improvement(self):
        agent = self.agent
        gamma = self.gamma

        new_policy = np.zeros_like(agent.pi)
        r = agent.r
        value_pi = agent.value_pi
        for s in range(agent.s_len):
            for act in range(agent.a_len):
                agent.value_q[s, act] = np.dot(agent.p[act, s, :], r + gamma * value_pi)
            # update policy
            idx = np.argmax(agent.value_q[s, :])
            new_policy[s, idx] = 1

        if np.all(np.equal(new_policy, agent.pi)):
            return False

        agent.pi = new_policy
        return True
