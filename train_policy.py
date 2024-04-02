import time

import numpy as np

from env import SnakeEnvV1
from algo.policy_based import PolicyGradient


def test(env, agent, name='algo', max_len=100, **kwargs):
    state, done = env.reset(reuse=True, **kwargs), False
    return_val = 0.0
    step = 0

    while not done:
        # act = np.argmax(agent.play(state))
        act = agent.play(state)
        print(state, act)
        next_state, reward, done, _ = env.step(act, **kwargs)
        return_val += reward
        step += 1
        state = next_state
        env.render(mode=name + ': {}/{}'.format(step, max_len))
        if step >= max_len:
            break
    print('Total reward:', return_val, done)
    print('Total step:', step)
    return step, return_val, int(done)


def train(env, algo, max_len=100, **kwargs):
    print('\n------------------------------------------')
    env.reset(reuse=True, verbose=True)
    algo = algo(env, **kwargs)
    print('Algorithm: ', algo.name)
    start = time.time()
    algo.update()
    print('Time consumption: ', time.time() - start)
    test(env, algo.agent, name=algo.name, max_len=max_len)
    print('------------------------------------------')


def main():
    # Environment
    env = SnakeEnvV1(size=20, num_ladders=0, num_targets=1)
    # Parameters
    alpha = 0.001
    gamma = 0.99
    epsilon = 0.5
    max_len = 100
    improve_iter = 1000

    # algo: VPG
    kwargs = {
        'alpha': alpha,
        'gamma': gamma,
        'max_len': max_len,
        'improve_iter': improve_iter,
        'epsilon': epsilon
    }
    train(env, algo=PolicyGradient, **kwargs)

    env.close()


if __name__ == '__main__':
    main()
