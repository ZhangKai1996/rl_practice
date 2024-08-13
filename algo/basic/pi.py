import time

from tqdm import tqdm
import numpy as np

from common.rendering import save_animation
from algo.agent import PlanningAgent
from algo.misc import softmax


class PolicyIteration:
    def __init__(self, env, gamma=0.99, eval_iter=-1, improve_iter=-1, **kwargs):
        self.name = 'PI'
        self.gamma = gamma
        self.agent = PlanningAgent(env, **kwargs)
        self.e_iter = int(1e6) if eval_iter <= 0 else eval_iter
        self.i_iter = int(1e6) if improve_iter <= 0 else improve_iter
        self.v_lst = []
        print('Algorithm: ', self.name)

    def evaluate_pi(self):
        """
        .. math:: V = \\Pi P (R + \\gamma V)
        .. math:: (I - \\gamma \\Pi P) V = \\Pi P R \\Leftrightarrow ax = b
        .. math:: \\Rightarrow a = I - \\gamma \\Pi P
        .. math:: \\Rightarrow b = \\Pi P R
        """
        agent = self.agent
        num_obs, num_act = agent.num_obs, agent.num_act
        p = agent.p
        r = agent.r
        pi = agent.pi
        a, b = np.eye(num_obs), np.zeros((num_obs,))
        for s in range(num_obs):
            a[s, :] -= self.gamma * np.dot(pi[s, :], p[:, s, :])
            b[s] = np.dot(pi[s, :], np.dot(p[:, s, :], r))
        v = np.linalg.solve(a, b)
        return v

    def update(self, prefix=''):
        count, start = 0, time.time()
        for iteration in tqdm(range(self.i_iter), desc='Iteration'):
            count += self.__evaluation(max_iter=self.e_iter)
            if not self.__improvement():
                print('Iteration: {}({})'.format(iteration, count))
                print('Time consumption: ', time.time() - start)
                break
        # save_animation(values=self.v_lst, r=self.agent.r, algo=self.name)
        self.agent.visual(algo=self.name+'_'+prefix)
        return self.agent

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
            self.v_lst.append(new_v.copy())

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
            q = np.dot(agent.p[:, s, :], r + self.gamma * v)
            agent.q[s, :] = q[:]
            ids = np.argwhere(q == q.max()).squeeze(axis=1)
            for idx in ids:
                new_policy[s, idx] = 1.0 / len(ids)
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        agent.pi = new_policy
        return True
