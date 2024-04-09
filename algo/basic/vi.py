import numpy as np

from algo.agent import PlanningAgent


class ValueIteration:
    def __init__(self, env, gamma=0.99, eval_iter=-1, improve_iter=-1, **kwargs):
        self.name = 'VI'
        self.gamma = gamma
        self.agent = PlanningAgent(env, **kwargs)
        self.e_iter = int(1e6) if eval_iter <= 0 else eval_iter
        self.i_iter = int(1e6) if improve_iter <= 0 else improve_iter

    def update(self):
        iteration = 0
        while True:
            iteration += 1
            self.__evaluation(max_iter=1)
            if not self.__improvement() or iteration >= self.i_iter:
                break

        print('Iteration: ', iteration)
        self.agent.visual(algo=self.name)
        return self.agent.pi, iteration

    def __evaluation(self, max_iter):
        agent = self.agent
        count = 0
        while True:
            old_v = agent.v
            new_v = old_v.copy()
            r = agent.r
            for s in range(agent.num_obs):
                q = np.dot(agent.p[:, s, :], r + self.gamma * old_v)
                new_v[s] = q.max()
            agent.v = new_v.copy()
            count += 1
            diff = np.sqrt(np.sum(np.power(old_v - new_v, 2)))
            if diff < 1e-6 or count >= max_iter:
                break

    def __improvement(self):
        agent = self.agent
        new_policy = np.zeros_like(agent.pi)
        v = agent.v
        r = agent.r
        for s in range(agent.num_obs):
            for a in range(agent.num_act):
                agent.q[s, a] = np.dot(agent.p[a, s, :], r + self.gamma * v)
            idx = np.argmax(agent.q[s, :])
            new_policy[s, idx] = 1.0
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        agent.pi = new_policy
        return True
