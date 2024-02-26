import sys
import logging
import itertools

import numpy as np

np.random.seed(0)
import gym
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    stream=sys.stdout, datefmt='%H:%M:%S')
env = gym.make('Taxi-v3')
for key in vars(env):
    logging.info('%s: %s', key, vars(env)[key])
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])


class QLearningAgent:
    def __init__(self, env):
        self.gamma = 0.99
        self.learning_rate = 0.2
        self.epsilon = 0.01
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []

    def step(self, observation, reward, terminated):
        if self.mode == 'train' and np.random.uniform() < self.epsilon:
            action = np.random.randint(self.action_n)
        else:
            action = self.q[observation].argmax()
        if self.mode == 'train':
            self.trajectory += [observation, reward, terminated, action]
            if len(self.trajectory) >= 8:
                self.learn()
        return action

    def close(self):
        pass

    def learn(self):
        state, _, _, action, next_state, reward, terminated, _ = \
            self.trajectory[-8:]

        v = reward + self.gamma * self.q[next_state].max() * (1. - terminated)
        target = reward + self.gamma * v * (1. - terminated)
        td_error = target - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error


agent = QLearningAgent(env)


def play_episode(env, agent, seed=None, mode=None, render=False):
    observation, _ = env.reset(seed=seed)
    reward, terminated, truncated = 0., False, False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.step(observation, reward, terminated)
        if render:
            env.render()
        if terminated or truncated:
            break
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
    agent.close()
    return episode_reward, elapsed_steps


logging.info('==== train ====')
episode_rewards = []
for episode in itertools.count():
    episode_reward, elapsed_steps = play_episode(env, agent, seed=episode,
                                                 mode='train')
    episode_rewards.append(episode_reward)
    logging.info('train episode %d: reward = %.2f, steps = %d',
                 episode, episode_reward, elapsed_steps)
    if np.mean(episode_rewards[-200:]) > 8:
        break
plt.plot(episode_rewards)
plt.show()

logging.info('==== test ====')
episode_rewards = []
for episode in range(100):
    episode_reward, elapsed_steps = play_episode(env, agent)
    episode_rewards.append(episode_reward)
    logging.info('test episode %d: reward = %.2f, steps = %d',
                 episode, episode_reward, elapsed_steps)
logging.info('average episode reward = %.2f ± %.2f',
             np.mean(episode_rewards), np.std(episode_rewards))

env.close()
