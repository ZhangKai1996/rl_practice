import sys
import logging
import itertools

import gym
import numpy as np
import matplotlib.pyplot as plt


class SARSALambdaAgent:
    def __init__(self, env):
        self.gamma = 0.99
        self.learning_rate = 0.1
        self.epsilon = 0.01
        self.lambd = 0.6
        self.beta = 1.
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

        self.mode = None
        self.trajectory = []

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []
            self.e = np.zeros(self.q.shape)

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
        state, _, _, action, next_state, reward, terminated, next_action = \
            self.trajectory[-8:]

        # update eligibility trace
        self.e *= (self.lambd * self.gamma)
        self.e[state, action] = 1. + self.beta * self.e[state, action]

        # update value
        target = reward + self.gamma * self.q[next_state, next_action] * (1. - terminated)
        td_error = target - self.q[state, action]
        self.q += self.learning_rate * self.e * td_error


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

    agent = SARSALambdaAgent(env)

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

    env.close()


if __name__ == '__main__':
    main()
