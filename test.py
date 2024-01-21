import time

import numpy as np

from env import SnakeEnv
from algo import PolicyIteration


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
    env = SnakeEnv(size=20, num_targets=1)
    env.reset()
    print('\n-------------Policy Iteration-------------')
    start = time.time()
    pi = PolicyIteration(env).update()
    print('Time consumption: ', time.time()-start)
    test(env, pi, algo='Policy Iteration')
    print('------------------------------------------')
    env.close()


if __name__ == '__main__':
    train()
