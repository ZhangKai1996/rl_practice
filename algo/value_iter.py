import time
import numpy as np

from .agent import TableAgent


class ValueIteration(object):
    def __init__(self, env, eval_max_iter=-1):
        self.agent = TableAgent(env)
        self.eval_max_iter = eval_max_iter
        self.gamma = 0.8

    def update(self):
        start = time.time()
        agent = self.agent
        gamma = self.gamma
        max_iter = self.eval_max_iter

        count = 0
        while True:
            old_value_pi = agent.value_pi
            new_value_pi = np.zeros_like(old_value_pi)
            r = agent.r
            for s in range(agent.s_len):  # for each state
                p_act = agent.pi[s]
                value_sas = []
                for act in range(agent.a_len):  # for each act
                    prob = p_act[act]
                    value_sa = prob * np.dot(agent.p[act, s, :], r + old_value_pi)
                    value_sas.append(value_sa)
                new_value_pi[s] = max(value_sas)
            diff = np.sqrt(np.sum(np.power(old_value_pi - new_value_pi, 2)))
            if diff < 1e-6 or count >= max_iter:
                break
            agent.value_pi = new_value_pi

        new_policy = np.zeros_like(agent.pi)
        for s in range(agent.s_len):
            for act in range(agent.a_len):
                agent.value_q[s, act] = np.dot(agent.p[act, s, :], agent.r + gamma * agent.value_pi)
            idx = np.argmax(agent.value_q[s, :])
            new_policy[s, idx] = 1
        agent.pi = new_policy

        print('Time consumption:', time.time()-start)
        return agent.pi
