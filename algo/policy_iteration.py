import numpy as np

from .agent import PlanningAgent


class PolicyIteration(object):
    def __init__(self, env):
        self.agent = PlanningAgent(env, algo='PI')
        self.gamma = 0.95

    def update(self, max_eval_iter=-1):
        if max_eval_iter <= 0:
            max_eval_iter = int(1e6)

        iteration = 0
        while True:
            iteration += 1
            self.__evaluation(max_iter=max_eval_iter)
            if not self.__improvement():
                break
        print('Iteration: ', iteration)
        self.agent.visual(show=True)
        return self.agent.pi

    def __evaluation(self, max_iter):
        agent = self.agent

        count = 0
        while True:
            old_state_value = agent.state_value
            new_state_value = old_state_value.copy()
            r = agent.r
            for s in range(agent.s_len):
                p_act = agent.pi[s]
                new_value = 0.0
                for a in range(agent.a_len):
                    prob = p_act[a]
                    new_value += prob * np.dot(agent.p[a, s, :], r + self.gamma * old_state_value)
                new_state_value[s] = new_value
            count += 1
            diff = np.sqrt(np.sum(np.power(old_state_value - new_state_value, 2)))
            if diff < 1e-6 or count >= max_iter:
                break
            agent.state_value = new_state_value

    def __improvement(self):
        agent = self.agent
        new_policy = np.zeros_like(agent.pi)
        state_value = agent.state_value
        r = agent.r
        for s in range(agent.s_len):
            for a in range(agent.a_len):
                agent.state_action_value[s, a] = np.dot(agent.p[a, s, :], r + self.gamma * state_value)
            idx = np.argmax(agent.state_action_value[s, :])
            new_policy[s, idx] = 1.0
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        agent.pi = new_policy
        return True












