import time
import numpy as np

from env import SnakeDiscreteEnv
from algo.basic import *


def test(env, policy, name='algo', max_len=100, **kwargs):
    state, done = env.reset(reuse=True, **kwargs), False
    return_val = 0.0
    step = 0
    while not done:
        # act = np.argmax(policy[state])
        act_prob = policy[state]
        acts = np.argwhere(act_prob == act_prob.max())
        acts = acts.squeeze(axis=1)
        np.random.shuffle(acts)
        act = acts[0]

        next_state, reward, done, _ = env.step(act, verbose=True, **kwargs)
        return_val += reward
        step += 1
        state = next_state
        # env.render(mode=name + ': {}/{}'.format(step, max_len))
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
    pi_star, c_iter, c_count = algo.update()
    delta = time.time() - start
    print('Time consumption: ', delta)
    result = test(env, pi_star, name=algo.name, max_len=max_len)
    print('------------------------------------------')
    return pi_star, [c_iter, c_count, delta, ] + list(result[1:])


def policy_similarity(p1, p2):
    return int(np.all(p1 == p2))


def main(size=30, ladders=0, targets=1, obstacles=50):
    # Environment
    env = SnakeDiscreteEnv(
        size=size,
        num_ladders=ladders,
        num_targets=targets,
        num_obstacles=obstacles
    )
    # Parameters
    kwargs = {
        'gamma': 0.95,
        'max_len': 100,
        'eval_iter': 128,
        'improve_iter': 1000,
        'rew': 0
    }
    # Algo: PI
    pis = []
    results = []
    reward_settings = [0.1, 0.5, ] + list(range(1, 16))
    for rew_s in reward_settings:
        kwargs['rew'] = rew_s
        pi_, result = run(env, algo=PolicyIteration, **kwargs)
        pis.append(pi_)
        results.append(result)
    env.close()

    pi_ol = np.zeros(shape=(len(pis), len(pis)))
    for i, p1 in enumerate(pis):
        for j, p2 in enumerate(pis):
            pi_ol[i, j] = policy_similarity(p1, p2)
    return pi_ol, results


if __name__ == '__main__':
    num_iter = 100
    size_ = 30
    num_ladders = 0
    num_targets = 1
    num_obstacles = 90

    # main(size_, num_ladders, num_targets, num_obstacles)

    pi_overlap_array = []
    result_array = []
    for episode in range(num_iter):
        print('{}/{}'.format(episode + 1, num_iter))
        pi_ol, results = main(size_, num_ladders, num_targets, num_obstacles)
        pi_overlap_array.append(pi_ol)
        result_array.append(results)
    pi_overlap_array = np.array(pi_overlap_array)
    delta_array = np.array(result_array)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    data = pi_overlap_array.mean(axis=0)
    im = plt.imshow(data, cmap='hot')
    plt.colorbar(im)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, round(data[i, j], 2),
                     ha="center", va="center", color="blue")
    plt.savefig('figs/record_{}_{}_{}_{}_{}_1.pdf'.format(
        num_iter, size_, num_ladders, num_targets, num_obstacles
    ))

    fig, axes = plt.subplots(4, 1)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图间距

    dones, pre_array = [], None
    labels, iters, counts, returns = [], [], [], []
    for k in range(delta_array.shape[1]):
        array = delta_array[:, k]
        dones.append(round(array[:, 4].mean() * 100, 2))
        if k == 0:
            pre_array = array.copy()
            continue
        labels.append('Pi_{}-Pi_1'.format(k + 1))
        iters.append(array[:, 0] - pre_array[:, 0])
        counts.append(array[:, 1] - pre_array[:, 1])
        returns.append(array[:, 3] - pre_array[:, 3])
    print(dones)
    scales = list(range(len(labels)))
    axes[0].set_title('Success Rate')
    axes[0].plot(dones)
    mean_return = [round(v, 2) for v in np.mean(returns, axis=1)]
    axes[1].set_title('Mean of Return Delta')
    axes[1].plot(scales, mean_return)
    mean_iter = [round(v, 2) for v in np.mean(iters, axis=1)]
    axes[2].set_title('Mean of Convergence Delta')
    axes[2].plot(scales, mean_iter)
    mean_count = [round(v, 2) for v in np.mean(counts, axis=1)]
    axes[3].set_title('Mean of Count Delta')
    axes[3].plot(scales, mean_count)
    plt.savefig('figs/record_{}_{}_{}_{}_{}_2.pdf'.format(
        num_iter, size_, num_ladders, num_targets, num_obstacles
    ))
    plt.show()
