from tqdm import tqdm
import numpy as np

from algo.agent import LearningAgent


class MonteCarlo:
    def __init__(self, env, gamma=0.99, epsilon=0.0, alpha=1e-2, max_len=100,
                 eval_iter=-1, improve_iter=-1, **kwargs):
        self.env = env
        self.name = 'MC'
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_len = max_len
        self.agent = LearningAgent(env)
        self.e_iter = int(1e6) if eval_iter <= 0 else eval_iter
        self.i_iter = int(1e6) if improve_iter <= 0 else improve_iter

    def update(self):
        for i in tqdm(range(self.i_iter), desc='MC update:'):
            for _ in range(self.e_iter):
                self.__evaluation()
            if not self.__improvement():
                print('Iteration: ', i+1)
                break
        self.agent.visual(algo=self.name)
        return self.agent

    def __evaluation(self):
        state = self.env.reset(reuse=True)
        experiences = []
        while True:
            action = self.agent.play(state, self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            experiences.append((state, action, reward))
            state = next_state
            if done or len(experiences) >= self.max_len:
                break
        values = []
        return_val = 0
        for (s, a, r) in reversed(experiences):
            return_val = return_val * self.gamma + r
            values.append((s, a, return_val))
        for (s, a, return_val) in reversed(values):
            self.agent.n[s, a] += 1
            diff = return_val - self.agent.q[s, a]
            self.agent.q[s, a] += diff / self.agent.n[s, a]

    def __improvement(self):
        new_policy = np.zeros_like(self.agent.pi)
        for s in range(self.agent.num_obs):
            # idx = np.argmax(self.agent.q[s, :])
            # new_policy[s, idx] = 1.0
            q_s = self.agent.q[s, :]
            ids = np.argwhere(q_s == q_s.max())
            ids = ids.squeeze(axis=1)
            for idx in ids:
                new_policy[s, idx] = 1.0 / len(ids)

        if np.all(np.equal(new_policy, self.agent.pi)):
            return False
        self.agent.pi = new_policy
        return True
