import time

import numpy as np

from env import SnakeDiscreteEnv
from algo.value_based import MonteCarlo, SARSA, QLearning


def test(env, agent, name='algo', max_len=100, **kwargs):
    state, done = env.reset(reuse=True, **kwargs), False
    return_val = 0.0
    step = 0

    while not done:
        act = agent.play(state, epsilon=0.0)
        next_state, reward, done, _ = env.step(act, verbose=True, **kwargs)
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
    agent = algo.update()
    print('Time consumption: ', time.time() - start)
    test(env, agent, name=algo.name, max_len=max_len)
    print('------------------------------------------')


def main():
    # Environment
    env = SnakeDiscreteEnv(size=20, num_ladders=0, num_targets=5, num_obstacles=0)
    # Parameters
    alpha = 0.01
    gamma = 0.95
    epsilon = 0.8
    max_len = 100
    eval_iter = 512
    improve_iter = 1000

    # algo: MC, TD
    kwargs = {
        'lamb': 0.0,
        'alpha': alpha,
        'gamma': gamma,
        'max_len': max_len,
        'eval_iter': eval_iter,
        'improve_iter': improve_iter,
        'epsilon': epsilon
    }
    train(env, algo=MonteCarlo, **kwargs)
    # train(env, algo=SARSA, **kwargs)
    # train(env, algo=QLearning, **kwargs)

    env.close()


if __name__ == '__main__':
    main()
