import time
import numpy as np

from env import SnakeEnv
from algo.basic import *


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


def run(env, algo, max_len=100, **kwargs):
    print('\n------------------------------------------')
    env.reset(reuse=True, verbose=True)
    algo = algo(env, **kwargs)
    print('Algorithm: ', algo.name)
    start = time.time()
    pi_star = algo.update()
    print('Time consumption: ', time.time() - start)
    test(env, pi_star, name=algo.name, max_len=max_len)
    print('------------------------------------------')


def main():
    # Environment
    env = SnakeEnv(size=20, num_ladders=30, num_targets=1)
    # Parameters
    kwargs = {
        'gamma': 0.95,
        'max_len': 100,
        'eval_iter': 128,
        'improve_iter': 1000
    }
    # Algo: BE
    run(env, algo=BellmanEquation, **kwargs)
    # Algo: PI
    run(env, algo=PolicyIteration, **kwargs)
    # Algo: VI
    run(env, algo=ValueIteration, **kwargs)
    env.close()


if __name__ == '__main__':
    main()
