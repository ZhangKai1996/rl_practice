import time

import numpy as np
import matplotlib.pyplot as plt

from env import SnakeEnv
from algo import MonteCarlo, SARSA, QLearning

from train import test


def train(**kwargs):
    env = SnakeEnv(**kwargs)
    env.reset()

    eval_iter, improve_iter = 32, 1000

    print('\n---------------Monte Carlo---------------')
    env.reset(reuse=True)
    start = time.time()
    pi = MonteCarlo(env, epsilon=0.5, max_len=100).update(
        eval_iter=eval_iter,
        improve_iter=improve_iter,
        verbose=False
    )
    print('Time consumption: ', time.time() - start)
    r_mc = test(env, pi, algo='Monte Carlo', verbose=False)
    print('-----------------------------------------')

    print('\n---------------SARSA---------------')
    env.reset(reuse=True)
    start = time.time()
    pi = SARSA(env, epsilon=0.5, max_len=100, lamd=0.0).update(
        eval_iter=eval_iter,
        improve_iter=improve_iter,
        verbose=False
    )
    print('Time consumption: ', time.time() - start)
    r_sarsa = test(env, pi, algo='SARSA', verbose=False)
    print('-------------------------------------')

    print('\n---------------Q Learning---------------')
    env.reset(reuse=True)
    start = time.time()
    pi = QLearning(env, epsilon=0.5, max_len=100).update(
        eval_iter=eval_iter,
        improve_iter=improve_iter,
        verbose=False
    )
    print('Time consumption: ', time.time() - start)
    r_ql = test(env, pi, algo='Q Learning', verbose=False)
    print('-----------------------------------------')

    env.close()

    return [
        r_mc[0], r_sarsa[0], r_ql[0],
        r_mc[1], r_sarsa[1], r_ql[1],
        r_mc[2], r_sarsa[2], r_ql[2],
    ]


def plot(record_array):
    print(record_array.shape)
    labels = ["MC", "SARSA", "QL"]

    fig, axes = plt.subplots(3, 2)

    # Plot Step
    rec_step_array = record_array[:, :3]

    max_step = (int(np.max(rec_step_array) // 5) + 1) * 5
    if max_step > 50:
        num_step = 20
    elif max_step <= 5:
        num_step = 5
    else:
        num_step = 10
    bins_step = np.linspace(0, max_step, num_step)
    print(bins_step)

    for i, label in enumerate(labels):
        ax = axes[i, 0]
        array = rec_step_array[:, i]
        ax.hist(array, bins=bins_step, label=label)
        ax.tick_params(labelsize=13)
        ax.legend()

    # Plot Reward
    rec_rew_array = record_array[:, 3:6]
    min_v, max_v = int(np.min(rec_rew_array)) - 1, int(np.max(rec_rew_array)) + 1
    bins_rew = np.linspace(min_v, max_v + 1, 20)

    rec_sr_array = record_array[:, 6:].mean(axis=0)
    for i, label in enumerate(labels):
        ax = axes[i, 1]
        array = rec_rew_array[:, i]
        ax.hist(array, bins=bins_rew, label=label)
        ax.text(0.5, 0.5, str(rec_sr_array[i]))
        ax.tick_params(labelsize=13)
        ax.legend()

    plt.savefig('fig_{}_{}.png'.format(size, num_targets))
    plt.show()


if __name__ == '__main__':
    size = 20
    num_targets = 3

    records = []
    for _ in range(1):
        records.append(train(size=size, num_targets=num_targets))
    plot(record_array=np.array(records))
