import time
import numpy as np

from env import SnakeEnv
from algo.basic import *


def test(env, policy, name='algo', max_len=100, **kwargs):
    state, done = env.reset(reuse=True, **kwargs), False
    return_val = 0.0
    step = 0

    while not done:
        # act = np.argmax(policy[state])
        act_prob = policy[state]
        acts = np.argwhere(act_prob == act_prob.max())
        print('\t>>>', acts.shape, end=' ')
        acts = acts.squeeze(axis=1)
        print(act_prob, acts, end=' ')
        np.random.shuffle(acts)
        print(acts[0])
        next_state, reward, done, _ = env.step(acts[0], **kwargs)
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
    delta = time.time() - start
    print('Time consumption: ', delta)
    result = test(env, pi_star, name=algo.name, max_len=max_len)
    print('------------------------------------------')
    return pi_star, [delta, ] + list(result[1:])


def main(size=30, ladders=0, targets=1, obstacles=50):
    # Environment
    env = SnakeEnv(size=size, num_ladders=ladders, num_targets=targets, num_obstacles=obstacles)
    # Parameters
    kwargs = {
        'gamma': 0.95,
        'max_len': 100,
        'rew': 0
    }
    # Algo: BE
    pi_1, result_1 = run(env, algo=BellmanEquation, **kwargs)
    # Algo: BE
    kwargs['rew'] = 1
    pi_2, result_2 = run(env, algo=BellmanEquation, **kwargs)
    env.close()
    return [int(np.all(pi_1 == pi_2)), ] + result_1 + result_2


if __name__ == '__main__':
    num_iter = 100
    size_ = 30
    num_ladders = 0
    num_targets = 1
    num_obstacles = 50

    delta_array = []
    for episode in range(num_iter):
        print('{}/{}'.format(episode+1, num_iter))
        delta_array.append(main(size_, num_ladders, num_targets, num_obstacles))
    delta_array = np.array(delta_array)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1)
    o_rate = round(delta_array[:, 0].mean()*100, 2)
    d_rate_1 = round(delta_array[:, 3].mean()*100, 2)
    d_rate_2 = round(delta_array[:, 6].mean()*100, 2)
    delta_tim = delta_array[:, 1] - delta_array[:, 4]
    axes[0].set_title('Overlap: {}%, Done: ({}%,{}%), Mean: {}'.format(
        o_rate, d_rate_1, d_rate_2, round(np.mean(delta_tim), 2)
    ))
    axes[0].hist(delta_tim, label='Pi_1-Pi_2')
    axes[0].legend()
    delta_ret = delta_array[:, 2] - delta_array[:, 5]
    axes[1].set_title('Mean of Return Delta: {}'.format(round(np.mean(delta_ret), 2)))
    axes[1].hist(delta_ret, label='Ret_1-Ret_2')
    axes[1].legend()
    plt.savefig('figs/record_{}_{}_{}_{}_{}.pdf'.format(
        num_iter, size_, num_ladders, num_targets, num_obstacles
    ))
    plt.show()
