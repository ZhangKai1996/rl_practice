from tqdm import tqdm
import numpy as np

from .agent import LearningAgent


class MonteCarlo(object):
    def __init__(self, env, epsilon=0.0, max_len=-1):
        self.env = env
        self.epsilon = epsilon
        self.max_len = max_len
        self.agent = LearningAgent(env, algo='MC')
        self.gamma = 0.95
        self.alpha = 0.01

    def update(self, eval_iter=32, improve_iter=1000, **kwargs):
        for i in tqdm(range(improve_iter), desc='MC update:'):
            for _ in range(eval_iter):
                self.__evaluation(**kwargs)

            if not self.__improvement():
                print('Iteration: ', i+1)
                break

        self.agent.visual()
        return self.agent.pi

    def __evaluation(self, **kwargs):
        state = self.env.reset(reuse=True, **kwargs)

        experiences = []
        while True:
            action = self.agent.play(state, self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            experiences.append((state, action, reward))
            state = next_state

            if done or len(experiences) >= self.max_len:
                break

        value = []
        return_val = 0
        for (s, a, r) in reversed(experiences):
            return_val = return_val * self.gamma + r
            value.append((s, a, return_val))

        for (s, a, return_val) in reversed(value):
            self.agent.value_n[s, a] += 1
            self.agent.value_q[s, a] += (return_val - self.agent.value_q[s, a]) / self.agent.value_n[s, a]

    def __improvement(self):
        new_policy = np.zeros_like(self.agent.pi)

        for s in range(self.agent.s_len):
            idx = np.argmax(self.agent.value_q[s, :])
            new_policy[s, idx] = 1.0

        if np.all(np.equal(new_policy, self.agent.pi)):
            return False

        self.agent.pi = new_policy
        return True
