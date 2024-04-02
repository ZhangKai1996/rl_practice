import csv

from tqdm import tqdm
import numpy as np

from algo.agent import LearningAgent


class QLearning(object):
    def __init__(self, env, gamma=0.99, epsilon=0.0, alpha=1e-2, max_len=100,
                 eval_iter=-1, improve_iter=-1, **kwargs):
        self.env = env
        self.name = 'QL'
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_len = max_len
        self.agent = LearningAgent(env)
        self.e_iter = int(1e6) if eval_iter <= 0 else eval_iter
        self.i_iter = int(1e6) if improve_iter <= 0 else improve_iter

    def update(self):
        losses = []
        for i in tqdm(range(self.i_iter), desc='Q Learning update: '):
            outputs = []
            for _ in range(self.e_iter):
                outputs.append(self.__evaluation())
            outputs = np.array(outputs)
            losses.append(list(np.mean(outputs, axis=0)))
            if not self.__improvement():
                print('Iteration: ', i+1)
                break
            self.agent.visual(algo=self.name)

        with open('figs/q_loss.csv', 'w', newline='') as f:
            f = csv.writer(f)
            f.writerows(losses)
        return self.agent.pi

    def __evaluation(self):
        s = self.env.reset(reuse=True)

        step = 0
        loss, reward_epi = [], 0.0
        while True:
            a = self.agent.play(s, self.epsilon)
            s_prime, r, done, _ = self.env.step(a)
            reward_epi += r
            # self.env.render(show=True)
            loss.append(self.learn(s, a, r, s_prime, done))
            s = s_prime
            step += 1
            if done or step >= self.max_len:
                break
        return [np.mean(loss), reward_epi]

    def learn(self, s, a, r, s_prime, done):
        q_predict = self.agent.q[s, a]
        q_real = r + self.gamma * np.max(self.agent.q[s_prime, :]) * (1-int(done))
        td_error = q_real - q_predict
        self.agent.q[s, a] += self.alpha * td_error
        return td_error

    def __improvement(self):
        new_policy = np.zeros_like(self.agent.pi)
        for s in range(self.agent.num_obs):
            idx = np.argmax(self.agent.q[s, :])
            new_policy[s, idx] = 1.0

        if np.all(np.equal(new_policy, self.agent.pi)):
            return False
        self.agent.pi = new_policy
        return True
