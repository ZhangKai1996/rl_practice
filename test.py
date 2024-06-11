import time
import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt

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
        env.render(mode=name + ': {}/{}'.format(step, max_len))
        if step >= max_len:
            break
    print('Total reward:', return_val)
    print('Total step:', step)
    return step, return_val, int(done)


def transition(node_dict, state, policy, env, step, max_len):
    if state in node_dict.keys():
        return

    node_dict[state] = []
    act_prob = policy[state]
    acts = np.argwhere(act_prob == act_prob.max())
    for act in acts.squeeze(axis=1):
        new_state = env.execute_action(act, state)
        if new_state == state:
            continue
        node_dict[state].append({'a': act, 's_prime': new_state, 'prob': act_prob[act]})
        if new_state in env.targets:
            return
        transition(node_dict, new_state, policy, env, step + 1, max_len)
    if step >= max_len:
        return


def optimal_paths(env, policy, max_len=100, **kwargs):
    state = env.reset(reuse=True, **kwargs)
    node_dict = {}
    transition(node_dict, state, policy, env, 0, max_len)
    return node_dict


def policy_comp(p1, p2, env):
    size = env.size
    sim_array = np.ones(shape=(size, size))
    p1_array = np.zeros(shape=(size, size))
    p2_array = np.zeros(shape=(size, size))
    for idx, act_prob1 in enumerate(p1):
        # if idx in env.obstacles:
        #     continue
        act_prob2 = p2[idx]
        i, j = idx // size, idx % size
        sim_array[i, j] = np.all(act_prob1 == act_prob2)

        # acts = np.argwhere(act_prob1 == act_prob1.max()).squeeze(axis=1)
        acts = list(np.int16(act_prob1 == act_prob1.max()))
        y = [x * 2 ** i for i, x in enumerate(acts)]
        p1_array[i, j] = sum(y) / 2 ** len(acts)
        acts = list(np.int16(act_prob1 == act_prob2.max()))
        y = [x * 2 ** i for i, x in enumerate(acts)]
        p2_array[i, j] = sum(y) / 2 ** len(acts)
    return sim_array, p1_array, p2_array


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
        'max_len': int(1e3),
        'eval_iter': 128,
        'improve_iter': 1000,
        'rew': 0
    }
    # Algo: PI (rew=0)
    pi1, result1 = run(env, algo=PolicyIteration, **kwargs)
    node_dict1 = optimal_paths(env, pi1, **kwargs)
    # Algo: PI (rew=1)
    kwargs['rew'] = 1
    pi2, result2 = run(env, algo=PolicyIteration, **kwargs)
    node_dict2 = optimal_paths(env, pi2, **kwargs)

    policy_comp_array = policy_comp(pi1, pi2, env)
    env.close()
    return (
        {'demo1': node_dict1, 'demo2': node_dict2},
        int(np.all(pi1 == pi2)),
        int(node_dict1 == node_dict2),
        policy_comp_array,
        [result1[-1], result2[-1]]
    )


def plot_tree(node_dict, filename):
    # 画布
    g = Digraph('G', filename='figs/' + filename + '.gv')
    # 定义两个节点
    all_edges = []
    for i, (s, edges) in enumerate(node_dict.items()):
        name = str(s)
        g.node(name, label=name)
        for e in edges:
            g.edge(name, str(e['s_prime']), label='{}({:>4.2f})'.format(e['a'], e['prob']))
            all_edges.append(name + '-' + str(e['s_prime']))
    # 渲染
    g.view()
    return all_edges


def plot(datas):
    fig, axes = plt.subplots(len(datas), 1)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图间距

    for i, data in enumerate(datas):
        ax = axes[i]
        plt.colorbar(ax.imshow(data, cmap='hot'))
        # for i in range(data.shape[0]):
        #     for j in range(data.shape[1]):
        #         ax.text(j, i, round(data[i, j], 1), size=6,
        #                 ha="center", va="center", color="blue")
    plt.savefig('figs/record_{}_{}_{}_{}_{}.pdf'.format(
        num_iter, size_, num_ladders, num_targets, num_obstacles
    ))
    # plt.show()


if __name__ == '__main__':
    num_iter = 100
    size_ = 30
    num_ladders = 0
    num_targets = 1
    num_obstacles = 180

    result_array = []
    for episode in range(num_iter):
        print('{}/{}'.format(episode + 1, num_iter))
        two_node_dict, *result = main(size_, num_ladders, num_targets, num_obstacles)
        is_same_policy, is_same_path, datas_, *_ = result[:]
        if is_same_path:
            continue

        tmp = []
        for g_name, ret in two_node_dict.items():
            edges = plot_tree(ret, filename=g_name)
            tmp.append(edges)
        [edges1, edges2, *_] = tmp
        plot(datas_)
        result_array.append(
            [episode, is_same_policy, is_same_path,
             int(all([edge in edges2 for edge in edges1])),
             int(all([edge in edges1 for edge in edges2]))] +
            result[-1]
        )
    result_array = np.array(result_array)
    print(num_iter, len(result_array), result_array[:, 1:].mean(axis=0))
    print(result_array)
