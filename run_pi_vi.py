import time

import numpy as np
import matplotlib.pyplot as plt

from env import SnakeEnv
from algo import PolicyIteration, ValueIteration

from train import test


def train(**kwargs):
    env = SnakeEnv(**kwargs)

    print('\n-------------Policy Iteration-------------')
    env.reset()
    start = time.time()
    pi = PolicyIteration(env).update()
    print('Time consumption: ', time.time() - start)
    r_pi = test(env, pi, algo='Policy Iteration')
    print('------------------------------------------')

    print('\n-------------Value Iteration-------------')
    env.reset(reuse=True)
    start = time.time()
    pi = ValueIteration(env).update(max_eval_iter=10)
    print('Time consumption: ', time.time() - start)
    r_vi = test(env, pi, algo='Value Iteration')
    print('-----------------------------------------')

    env.close()

    return [
        r_pi[0], r_vi[0],
        r_pi[1], r_vi[1],
        r_pi[2], r_vi[2],
    ]


def plot(record_array):
    print(record_array.shape)
    labels = ["PI", "VI"]

    fig, axes = plt.subplots(2, 2)

    # Plot Step
    rec_step_array = record_array[:, :2]

    max_step = (int(np.max(rec_step_array) // 5) + 1) * 5
    if max_step > 50:
        num_step = 20
    elif max_step <= 5:
        num_step = 5
    else:
        num_step = 10
    bins_step = np.linspace(0, max_step, num_step+1)
    print(bins_step)

    for i, label in enumerate(labels):
        ax = axes[i, 0]
        array = rec_step_array[:, i]
        ax.hist(array, bins=bins_step, label=label)
        ax.tick_params(labelsize=13)
        ax.legend()

    # Plot Reward
    rec_rew_array = record_array[:, 2:4]
    min_v, max_v = int(np.min(rec_rew_array)) - 1, int(np.max(rec_rew_array)) + 1
    bins_rew = np.linspace(min_v, max_v + 1, 20)

    rec_sr_array = record_array[:, 4:].mean(axis=0)
    for i, label in enumerate(labels):
        ax = axes[i, 1]
        array = rec_rew_array[:, i]
        ax.hist(array, bins=bins_rew, label=label)
        ax.text(0.5, 0.5, str(rec_sr_array[i]))
        ax.tick_params(labelsize=13)
        ax.legend()

    plt.savefig('figs/fig_p_{}_{}.png'.format(size, num_targets))
    plt.show()


if __name__ == '__main__':
    size = 20
    num_targets = 2

    records = []
    for _ in range(1):
        records.append(train(size=size, num_targets=num_targets))
    plot(record_array=np.array(records))

