import csv

from tqdm import tqdm
import numpy as np

from algo.agent import NetworkAgent


class PolicyGradient(object):
    def __init__(self, env, gamma=0.99, epsilon=0.0, alpha=1e-2, max_len=100,
                 improve_iter=-1, **kwargs):
        self.env = env
        self.name = 'VPG'
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_len = max_len
        self.agent = NetworkAgent(env, lr=alpha)
        self.i_iter = int(1e6) if improve_iter <= 0 else improve_iter

    def update(self):
        logs = []
        for i in tqdm(range(self.i_iter), desc='VPG update:'):
            state = self.env.reset(reuse=True)

            experiences, rew_sum = [], 0.0
            while True:
                # print(state)
                action = self.agent.play(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                experiences.append([state, action, reward, float(done)])
                self.env.render(show=True)
                rew_sum += reward
                state = next_state
                if done or len(experiences) >= self.max_len:
                    break

            return_val = 0.0
            states, actions, rewards, dones = [], [], [], []
            for [s, a, r, d] in reversed(experiences):
                return_val = return_val * self.gamma + r
                states.append(s)
                actions.append(a)
                dones.append(d)
                rewards.append(return_val)

            loss = self.agent.learn(
                obs_batch=np.array(states[::-1]),
                act_batch=np.array(actions[::-1]),
                rew_batch=np.array(rewards[::-1]),
                done_batch=np.array(dones[::-1])
            )
            logs.append([i, rew_sum, loss])
            print(i, rew_sum, loss)
            # if self.epsilon >= 0.05:
            #     self.epsilon *= 0.95

        with open('figs/log_vpg.csv', 'w', newline='') as f:
            f = csv.writer(f)
            f.writerows(logs)
