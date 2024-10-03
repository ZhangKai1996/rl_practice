import sys
import logging
import itertools

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SARSAAgent:
    def __init__(self, env):
        self.gamma = 0.9
        self.learning_rate = 0.2
        self.epsilon = 0.01
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

        self.mode = None
        self.trajectory = []

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

    def learn(self):
        state, _, _, action, next_state, reward, terminated, next_action = \
            self.trajectory[-8:]

        target = reward + self.gamma * self.q[next_state, next_action] * (1. - terminated)
        td_error = target - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error

    def close(self):
        pass


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


def main():
    np.random.seed(0)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        stream=sys.stdout, datefmt='%H:%M:%S')

    env = gym.make('Taxi-v3')
    for key in vars(env):
        logging.info('%s: %s', key, vars(env)[key])
    for key in vars(env.spec):
        logging.info('%s: %s', key, vars(env.spec)[key])

    state, _ = env.reset()
    taxirow, taxicol, passloc, destidx = env.unwrapped.decode(state)
    logging.info('location of taxi = %s', (taxirow, taxicol))
    logging.info('location of passager = %s', env.unwrapped.locs[passloc])
    logging.info('location of destination = %s', env.unwrapped.locs[destidx])
    # env.render()

    env.step(0)
    agent = SARSAAgent(env)

    logging.info('==== train ====')
    episode_rewards = []
    for episode in itertools.count():
        episode_reward, elapsed_steps = play_episode(env, agent, seed=episode,
                                                     mode='train')
        episode_rewards.append(episode_reward)
        logging.info('train episode %d: reward = %.2f, steps = %d',
                     episode, episode_reward, elapsed_steps)
        if np.mean(episode_rewards[-200:]) > env.spec.reward_threshold:
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
    print(pd.DataFrame(agent.q))
    policy = np.eye(agent.action_n)[agent.q.argmax(axis=-1)]
    print(pd.DataFrame(policy))
    env.close()


if __name__ == '__main__':
    main()
