import time

import numpy as np

from env import SnakeEnv
from algo.basic import PolicyIteration


def test(env, policy, name='algo', max_len=100, **kwargs):
    state, done = env.reset(reuse=True, **kwargs), False
    return_val = 0.0
    step = 0

    while not done:
        act = np.argmax(policy[state])
        next_state, reward, done, _ = env.step(act, **kwargs)
        return_val += reward
        step += 1
        state = next_state
        env.render(mode=name + ': {}/{}'.format(step, max_len))
        if step >= max_len:
            break
    print('Total reward:', return_val)
    print('Total step:', step)
    return step, return_val, int(done)


def train(env, algo, max_len=100, **kwargs):
    print('\n------------------------------------------')
    env.reset(reuse=True, verbose=True)
    algo = algo(env, **kwargs)
    print('Algorithm: ', algo.name)
    start = time.time()
    pi = algo.update()
    delta = time.time() - start
    print('Time consumption: ', delta)
    test(env, pi, name=algo.name, max_len=max_len)
    print('------------------------------------------')
    return delta


def main():
    # Environment
    env = SnakeEnv(size=20, num_ladders=0, num_targets=1)
    # Parameters
    alpha = 0.01
    gamma = 0.95
    epsilon = 0.5
    max_len = 100
    eval_iter = 128
    improve_iter = 1000
    # algo: PI, VI

    delta_list = []

    kwargs = {
        'gamma': gamma,
        'max_len': max_len,
        'eval_iter': eval_iter,
        'improve_iter': improve_iter,
        'reward': 1
    }
    env.x, env.y = 10.0, -1.0
    delta = train(env, algo=PolicyIteration, **kwargs)
    delta_list.append(delta)

    kwargs = {
        'gamma': gamma,
        'max_len': max_len,
        'eval_iter': eval_iter,
        'improve_iter': improve_iter,
        'reward': 2
    }
    env.x, env.y = 10.0, -1.0
    delta = train(env, algo=PolicyIteration, **kwargs)
    delta_list.append(delta)

    kwargs = {
        'gamma': gamma,
        'max_len': max_len,
        'eval_iter': eval_iter,
        'improve_iter': improve_iter,
        'reward': 1
    }
    env.x, env.y = 1.0, -1.0
    delta = train(env, algo=PolicyIteration, **kwargs)
    delta_list.append(delta)

    kwargs = {
        'gamma': gamma,
        'max_len': max_len,
        'eval_iter': eval_iter,
        'improve_iter': improve_iter,
        'reward': 2
    }
    env.x, env.y = 1.0, -1.0
    delta = train(env, algo=PolicyIteration, **kwargs)
    delta_list.append(delta)

    env.close()
    return np.array(delta_list)


if __name__ == '__main__':
    delta_array = []
    for episode in range(100):
        print('{}/{}'.format(episode+1, 2))
        delta_array.append(main())
    delta_array = np.array(delta_array)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, sharex=True)
    for i in range(4):
        axes[i].hist(delta_array[:, i])

    plt.savefig('figs/record.pdf')
    plt.show()
