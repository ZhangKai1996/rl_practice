import numpy as np
import scipy

from algo.agent import PlanningAgent
from common.rendering import save_animation


class BellmanEquation(object):
    def __init__(self, env, gamma=0.95, **kwargs):
        self.name = 'BE'
        self.agent = PlanningAgent(env, **kwargs)
        self.gamma = gamma

    def evaluate(self, pi, draw=False):
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

        # ax = b
        a, b = np.eye(num_obs), np.zeros((num_obs,))
        for s in range(num_obs):
            a[s, :] -= self.gamma * np.dot(pi[s], p[:, s, :])
            b[s, :] = np.dot(pi[s], np.dot(p[:, s, :], r))
        v = np.linalg.solve(a, b)
        q = np.zeros((num_obs, num_act))
        for s in range(num_obs):
            q[s, :] = np.dot(p[:, s, :], r + self.gamma * v)
        if draw:
            save_animation(values=[v, ], r=self.agent.r, algo=self.name)
        return v, q

    def update(self):
        agent = self.agent
        num_obs, num_act = agent.num_obs, agent.num_act
        p = agent.p
        r = np.array(agent.r)

        a_ub = np.zeros((num_obs, num_act, num_obs))
        e = np.eye(num_obs)
        b_ub = np.zeros((num_obs, num_act))
        for a in range(num_act):
            a_ub[:, a, :] = self.gamma * p[a, :, :] - e
        for s in range(num_obs):
            b_ub[s, :] -= np.dot(p[:, s, :], r)

        result = scipy.optimize.linprog(
            c=np.ones(num_obs),
            A_ub=a_ub.reshape(-1, num_obs),
            b_ub=b_ub.reshape(-1),
            bounds=[(None, None), ] * num_obs,
            method='interior-point'
        )
        v = result.x
        q = r + self.gamma * np.dot(p, v)
        q = q.transpose(1, 0)
        agent.q = q
        agent.v = v

        new_policy = np.zeros_like(agent.pi)
        for s in range(agent.num_obs):
            # idx = np.argmax(q[s, :])
            # new_policy[s, idx] = 1.0
            q_s = q[s, :]
            ids = np.argwhere(q_s == q_s.max())
            ids = ids.squeeze(axis=1)
            for idx in ids:
                new_policy[s, idx] = 1.0 / len(ids)
        agent.pi = new_policy
        # self.agent.visual(algo=self.name)
        save_animation(values=[v, ], r=r, algo=self.name)
        return agent.pi,
