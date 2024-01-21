import time

import numpy as np

from env import SnakeEnv
from algo import (PolicyIteration, ValueIteration, MonteCarlo, SARSA, QLearning)


def test(env, policy, algo='algo', max_iter=100, **kwargs):
    state, done = env.reset(reuse=True, **kwargs), False
    return_val = 0.0
    step = 0

    while not done:
        act = np.argmax(policy[state])
        next_state, reward, done, _ = env.step(act, **kwargs)
        return_val += reward
        step += 1
        state = next_state

        env.render(mode=algo+': {}/{}'.format(step, max_iter))

        if step >= max_iter:
            break

    print('Total reward:', return_val)
    print('Total step:', step)
    return step, return_val, int(done)


def train():
    env = SnakeEnv(size=10, num_targets=1)
    env.reset()

    # print('\n-------------Policy Iteration-------------')
    # start = time.time()
    # pi = PolicyIteration(env).update()
    # print('Time consumption: ', time.time()-start)
    # test(env, pi, algo='Policy Iteration')
    # print('------------------------------------------')

    # print('\n-------------Value Iteration-------------')
    # env.reset(reuse=True)
    # start = time.time()
    # pi = ValueIteration(env).update(max_eval_iter=10)
    # print('Time consumption: ', time.time()-start)
    # test(env, pi, algo='Value Iteration')
    # print('-----------------------------------------')

    eval_iter, improve_iter = 32, 1000

    print('\n---------------Monte Carlo---------------')
    env.reset(reuse=True)
    start = time.time()
    pi = MonteCarlo(env, epsilon=0.5, max_len=100).update(
        eval_iter=eval_iter,
        improve_iter=improve_iter,
        verbose=False
    )
    print('Time consumption: ', time.time()-start)
    test(env, pi, algo='Monte Carlo', verbose=False)
    print('-----------------------------------------')

    print('\n---------------SARSA---------------')
    env.reset(reuse=True)
    start = time.time()
    pi = SARSA(env, epsilon=0.5, max_len=100, lamd=0.0).update(
        eval_iter=eval_iter,
        improve_iter=improve_iter,
        verbose=False
    )
    print('Time consumption: ', time.time()-start)
    test(env, pi, algo='SARSA', verbose=False)
    print('-------------------------------------')

    print('\n---------------Q Learning---------------')
    env.reset(reuse=True)
    start = time.time()
    pi = QLearning(env, epsilon=0.5, max_len=100).update(
        eval_iter=eval_iter,
        improve_iter=improve_iter,
        verbose=False
    )
    print('Time consumption: ', time.time()-start)
    test(env, pi, algo='Q Learning', verbose=False)
    print('-----------------------------------------')

    env.close()


if __name__ == '__main__':
    train()
