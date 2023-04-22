import numpy as np


class TableAgent(object):
    def __init__(self, env):
        self.s_len = s_len = env.observation_space.n
        self.a_len = a_len = env.action_space.n
        print(s_len, a_len)

        p = np.zeros([a_len, s_len, s_len], dtype=np.float)
        print('p:', p.shape)
        for act in range(a_len):
            for s in range(s_len):
                s_prime = env.execute_action(act, s)
                p[act, s, s_prime] = 1

        self.p = p
        self.r = [env.reward(s)[0] for s in range(s_len)]
        self.pi = np.ones((s_len, a_len)) * 1.0/a_len
        self.value_pi = np.zeros((s_len,))
        self.value_q = np.zeros((s_len, a_len))


class ModelFreeAgent(object):
    def __init__(self, env):
        self.s_len = s_len = env.observation_space.n
        self.a_len = a_len = env.action_space.n

        self.pi = np.ones((s_len, a_len)) * 1.0/a_len
        self.value_q = np.zeros((s_len, a_len))
        self.value_n = np.zeros((s_len, a_len))
        self.gamma = 0.8

    def play(self, s, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.a_len)
        else:
            return np.argmax(self.pi[s])
