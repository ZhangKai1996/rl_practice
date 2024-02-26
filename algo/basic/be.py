import numpy as np
import scipy

from algo.agent import PlanningAgent


class BellmanEquation(object):
    def __init__(self, env):
        self.agent = PlanningAgent(env)
        self.gamma = 0.95

    def evaluate(self, pi):
        agent = self.agent
        num_obs, num_act = agent.num_obs, agent.num_act
        p = agent.p
        r = agent.r

        # Ax = b
        A, b = np.eye(num_obs), np.zeros((num_obs,))
        for s in range(num_obs):
            p_act = pi[s]
            for a in range(num_act):
                prob = p_act[a]
                A[s, :] -= prob * self.gamma * p[a, s, :]
                b[s] += prob * np.dot(p[a, s, :], r)
        v = np.linalg.solve(A, b)

        q = np.zeros((num_obs, num_act))
        for s in range(num_obs):
            for a in range(num_act):
                q[s, a] = np.dot(p[a, s, :], r + self.gamma * v)
        return v, q

    def solve(self):
        agent = self.agent
        num_obs, num_act = agent.num_obs, agent.num_act
        P = agent.p

        p = np.zeros((num_obs, num_act, num_obs))
        r = np.zeros((num_obs, num_act))
        for s in range(num_obs - 1):
            for a in range(num_act):
                for prob, s_prime, reward, terminated in P[s][a]:
                    p[s, a, s_prime] += prob
                    r[s, a] += (reward * prob)

        v = scipy.optimize.linprog(
            c=np.ones(num_obs),
            A_ub=self.gamma * p.reshape(-1, num_obs) - np.repeat(np.eye(num_obs), num_act, axis=0),
            b_ub=-r.reshape(-1),
            bounds=[(None, None), ] * num_obs,
            method='interior-point'
        ).x
        q = r + self.gamma * np.dot(p, v)
        return v, q
