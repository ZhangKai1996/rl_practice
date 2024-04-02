import sys
import logging
import itertools

import numpy as np

import pandas as pd
import gym
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf

from tensorflow import keras
from tensorflow import nn
from tensorflow import optimizers
from tensorflow import losses
from tensorflow.keras import layers

np.random.seed(0)
tf.random.set_seed(0)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    stream=sys.stdout, datefmt='%H:%M:%S')
env = gym.make('CartPole-v0')
for key in vars(env):
    logging.info('%s: %s', key, vars(env)[key])
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])


class VPGAgent:
    def __init__(self, env):
        self.action_n = env.action_space.n
        self.gamma = 0.99
        self.policy_net = self.build_net(
            hidden_sizes=[],
            output_size=self.action_n,
            output_activation=nn.softmax,
            loss=losses.categorical_crossentropy
        )

    def build_net(self, hidden_sizes, output_size, activation=nn.relu, output_activation=None,
                  use_bias=False, loss=losses.mse, learning_rate=0.005):
        model = keras.Sequential()
        for hidden_size in hidden_sizes:
            model.add(layers.Dense(units=hidden_size, activation=activation, use_bias=use_bias))
        model.add(layers.Dense(units=output_size, activation=output_activation, use_bias=use_bias))
        model.compile(optimizer=optimizers.Adam(learning_rate), loss=loss)
        return model

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []

    def step(self, observation, reward, terminated):
        probs = self.policy_net.predict(observation[np.newaxis], verbose=0)[0]
        action = np.random.choice(self.action_n, p=probs)
        if self.mode == 'train':
            self.trajectory += [observation, reward, terminated, action]
        return action

    def close(self):
        if self.mode == 'train':
            self.learn()

    def learn(self):
        df = pd.DataFrame(np.array(self.trajectory, dtype=object).reshape(-1, 4),
                          columns=['state', 'reward', 'terminated', 'action'])
        df['discount'] = self.gamma ** df.index.to_series()
        df['discounted_reward'] = df['discount'] * df['reward']
        df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
        states = np.stack(df['state'])
        actions = np.eye(self.action_n)[df['action'].astype(int)]
        sample_weight = df[['discounted_return', ]].values.astype(float)
        self.policy_net.fit(states, actions, sample_weight=sample_weight, verbose=1)


agent = VPGAgent(env)


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
    if np.mean(episode_rewards[-10:]) > 195:
        break
plt.plot(episode_rewards)

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
