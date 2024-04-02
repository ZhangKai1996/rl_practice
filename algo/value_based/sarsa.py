import csv

from tqdm import tqdm
import numpy as np

from algo.agent import LearningAgent


class SARSA(object):
    def __init__(self, env, gamma=0.99, epsilon=0.0, lamd=0.0, beta=1., alpha=1e-2, max_len=100,
                 eval_iter=-1, improve_iter=-1, **kwargs):
        self.env = env
        self.name = 'SARSA'
        self.gamma = gamma
        self.alpha = alpha
        self.lamd = lamd
        self.beta = beta
        self.epsilon = epsilon
        self.max_len = max_len
        self.agent = LearningAgent(env)
        self.e = None
        self.e_iter = int(1e6) if eval_iter <= 0 else eval_iter
        self.i_iter = int(1e6) if improve_iter <= 0 else improve_iter

    def update(self):
        losses = []
        for i in tqdm(range(self.i_iter), desc='SARSA update: '):
            outputs = []
            for _ in range(self.e_iter):
                outputs.append(self.__evaluation())
            outputs = np.array(outputs)
            losses.append(list(np.mean(outputs, axis=0)))
            if not self.__improvement():
                print('Iteration: ', i + 1)
                break
            self.agent.visual(algo=self.name)

        with open('figs/sarsa_loss.csv', 'w', newline='') as f:
            f = csv.writer(f)
            f.writerows(losses)

        return self.agent.pi

    def __evaluation(self):
        state = self.env.reset(reuse=True)
        self.e = np.zeros(self.agent.q.shape)

        step = 0
        trajectory, loss, reward_epi = [], [], 0.0
        while True:
            action = self.agent.play(state, self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            reward_epi += reward
            if len(trajectory) > 0:
                loss.append(self.learn(trajectory, state, action))
            # self.env.render(show=True)
            trajectory.append([state, action, reward, done])
            state = next_state
            step += 1
            if done or step >= self.max_len:
                break
        return [np.mean(loss), reward_epi]

    def learn(self, trajectory, s_prime, a_prime, p=0.01):
        [s, a, r, done] = trajectory[-1]
        # update eligibility trace
        self.e *= self.lamd * self.gamma
        self.e[s, a] = 1. + self.beta * self.e[s, a]

        q_predict = self.agent.q[s, a]
        # v = self.agent.q[s_prime].mean() * p + self.agent.q[s_prime].max() * (1. - p)
        v = self.agent.q[s_prime, a_prime]
        q_real = r + self.gamma * v * (1-int(done))
        td_error = q_real - q_predict
        # self.agent.q += self.alpha * self.e * td_error
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
