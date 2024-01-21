from tqdm import tqdm
import numpy as np

from .agent import LearningAgent


class QLearning(object):
    def __init__(self, env, epsilon=0.0, max_len=100):
        self.env = env
        self.epsilon = epsilon
        self.max_len = max_len
        self.agent = LearningAgent(env, algo='QL')
        self.gamma = 0.99
        self.alpha = 0.01

    def update(self, eval_iter=32, improve_iter=1000, **kwargs):
        for i in tqdm(range(improve_iter), desc='Q Learning update: '):
            for _ in range(eval_iter):
                self.__evaluation(**kwargs)

            if not self.__improvement():
                print('Iteration: ', i+1)
                break

        self.agent.visual()
        return self.agent.pi

    def __evaluation(self, **kwargs):
        s = self.env.reset(reuse=True, **kwargs)

        step = 0
        while True:
            a = self.agent.play(s, self.epsilon)
            s_prime, r, done, _ = self.env.step(a)

            # TD learning
            value_q_predict = self.agent.value_q[s, a]
            if done:
                value_q_real = r
            else:
                value_q_real = r + self.gamma * np.max(self.agent.value_q[s_prime, :])
            td_error = value_q_real - value_q_predict
            self.agent.value_q[s, a] += self.alpha * td_error

            s = s_prime
            step += 1
            if done or step >= self.max_len:
                break

    def __improvement(self):
        new_policy = np.zeros_like(self.agent.pi)

        for s in range(self.agent.s_len):
            idx = np.argmax(self.agent.value_q[s, :])
            new_policy[s, idx] = 1.0

        if np.all(np.equal(new_policy, self.agent.pi)):
            return False

        self.agent.pi = new_policy
        return True
