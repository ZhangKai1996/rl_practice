import numpy as np

from .agent import ModelFreeAgent


class SARSA(object):
    def __init__(self, env, epsilon=0.0):
        self.epsilon = epsilon
        self.agent = ModelFreeAgent(env)
        self.env = env

    def update(self):
        for i in range(10):
            for j in range(2000):
                self.__evaluation()
            self.__improvement()

    def __evaluation(self):
        env = self.env
        agent = self.agent

        state = env.reset()
        prev_state = -1
        prev_act = -1
        while True:
            act = agent.play(state, self.epsilon)
            next_state, reward, done, _ = env.step(act)
            if prev_act != -1:
                return_val = reward + agent.gamma * (0 if done else agent.value_q[state][act])
                agent.value_n[prev_state][prev_act] += 1
                agent.value_q[prev_state][prev_act] += (return_val - agent.value_q[prev_state][prev_act]) / \
                                                        agent.value_n[prev_state][prev_act]

            prev_act = act
            prev_state = state
            state = next_state

            if done:
                break

    def __improvement(self):
        agent = self.agent

        new_policy = np.zeros_like(agent.pi)
        for i in range(1, agent.s_len):
            new_policy[i] = np.argmax(agent.value_q[i, :])

        if np.all(np.equal(new_policy, agent.pi)):
            return False
        agent.pi = new_policy
        return True
