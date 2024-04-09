import numpy as np

from algo.agent import PlanningAgent


class PolicyIteration:
    def __init__(self, env, gamma=0.99, eval_iter=-1, improve_iter=-1, **kwargs):
        self.name = 'PI'
        self.gamma = gamma
        self.agent = PlanningAgent(env, **kwargs)
        self.e_iter = int(1e6) if eval_iter <= 0 else eval_iter
        self.i_iter = int(1e6) if improve_iter <= 0 else improve_iter

    def update(self):
        iteration = 0
        count = 0
        while True:
            iteration += 1
            # self.agent.visual(algo=self.name)
            count += self.__evaluation(max_iter=self.e_iter)
            if not self.__improvement() or iteration >= self.i_iter:
                break

        print('Iteration: ', iteration)
        self.agent.visual(algo=self.name)
        return self.agent.pi, iteration, count

    def __evaluation(self, max_iter):
        agent = self.agent
        count = 0
        while True:
            old_v = agent.v
            new_v = old_v.copy()
            r = agent.r
            for s in range(agent.num_obs):
                q = np.dot(agent.p[:, s, :], r + self.gamma * old_v)
                new_v[s] = np.dot(agent.pi[s], q)
            agent.v = new_v.copy()

            count += 1
            diff = np.sqrt(np.sum(np.power(old_v - new_v, 2)))
            if diff < 1e-6 or count >= max_iter:
                break
        return count

    def __improvement(self):
        agent = self.agent

        new_policy = np.zeros_like(agent.pi)
        v = agent.v
        r = agent.r
        for s in range(agent.num_obs):
            q_s = agent.q[s, :] = np.dot(agent.p[:, s, :], r + self.gamma * v)
            # idx = np.argmax(q_s)
            # new_policy[s, idx] = 1.0
            ids = np.argwhere(q_s == q_s.max())
            ids = ids.squeeze(axis=1)
            for idx in ids:
                new_policy[s, idx] = 1.0 / len(ids)
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        agent.pi = new_policy
        return True
