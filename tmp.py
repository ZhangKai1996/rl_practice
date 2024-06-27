
import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt

from env import SnakeDiscreteEnv
from algo.basic import *
from algo.misc import regularize
from common.utils import extract_all_paths


def train_and_test(env, algo, max_len=100, **kwargs):
    print('\n------------------------------------------')
    # Train
    env.reset(reuse=True, verbose=True)
    algo = algo(env, **kwargs)
    agent = algo.update()

    # Test
    state, done = env.reset(reuse=True), False
    return_val, step = 0.0, 0
    while not done:
        action = agent.action_sample(state)
        next_state, reward, done, _ = env.step(action, verbose=True)
        return_val += reward
        step += 1
        state = next_state
        env.render(mode=algo.name + ': {}/{}'.format(step, max_len))
        if step >= max_len:
            break
    print('Total reward:', return_val)
    print('Total step:{}/{}'.format(step, max_len))

    # Optimal Paths
    node_dict = {}
    state = env.reset(reuse=True)
    transition(node_dict, state, agent.pi, env, 0, max_len)
    print('------------------------------------------')
    return agent, return_val, int(done), node_dict


def transition(node_dict, state, policy, env, step, max_len):
    if state in node_dict.keys():
        return

    node_dict[state] = {}
    act_prob = policy[state]
    acts = np.argwhere(act_prob == act_prob.max())
    for act in acts.squeeze(axis=1):
        new_state = env.execute_action(act, state)
        if new_state == state:
            continue
        node_dict[state][new_state] = {'a': act, 's_prime': new_state, 'prob': act_prob[act]}
        if new_state in env.targets:
            return
        transition(node_dict, new_state, policy, env, step + 1, max_len)
    if step >= max_len:
        return


def run(episode, size=30, ladders=0, targets=1, obstacles=50):
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
        'max_len': int(1e2),
        'eval_iter': 128,
        'improve_iter': 1000,
        'rew': 0
    }
    # Algo: PI (rew=0)
    agent1, return1, done1, node_dict1 = train_and_test(env, algo=PolicyIteration, **kwargs)
    # Algo: PI (rew=1)
    kwargs['rew'] = 1
    agent2, return2, done2, node_dict2 = train_and_test(env, algo=PolicyIteration, **kwargs)
    env.close()

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(regularize(agent1.v.reshape(-1)), label='v')
    axes[0].plot(regularize(agent1.r.reshape(-1)), label='r')
    axes[1].plot(regularize(agent2.v.reshape(-1)), label='v')
    axes[1].plot(regularize(agent2.r.reshape(-1)), label='r')
    [ax.legend() for ax in axes]
    plt.show()

    edges1, probs1 = plot_tree(node_dict1, agent1.p, filename='graph1_{}'.format(episode))
    edges2, probs2 = plot_tree(node_dict2, agent2.p, filename='graph2_{}'.format(episode))
    sim_array = policy_comp(agent1.pi, agent2.pi, env)
    plot_data(sim_array, {'graph1': probs1, 'graph2': probs2}, prefix=episode)
    return (
        episode,
        int(np.all(agent1.pi == agent2.pi)),
        int(node_dict1 == node_dict2),
        int(all([edge in edges2 for edge in edges1])),
        int(all([edge in edges1 for edge in edges2])),
        done1, done2
    )


def policy_comp(p1, p2, env):
    size = env.size
    sim_array = np.ones(shape=(size, size))
    for idx, act_prob1 in enumerate(p1):
        if idx in env.obstacles:
            continue
        act_prob2 = p2[idx]
        i, j = idx // size, idx % size
        sim_array[i, j] = np.all(act_prob1 == act_prob2)
    return sim_array


def plot_tree(node_dict, p, filename):
    g = Digraph('G', filename='figs/' + filename + '.gv')
    edges = []
    for start, info in node_dict.items():
        name = str(start)
        g.node(name, label=name)
        for end, v in info.items():
            g.edge(name, str(end), label='{}({:>4.2f})'.format(v['a'], v['prob']))
            edges.append((start, end))
    # 提取所有最优路径
    all_paths = extract_all_paths(edges)
    probs = []
    for path in all_paths:
        prob = 1.0
        for i, s1 in enumerate(path[:-1]):
            s2 = path[i+1]
            info = node_dict[s1][s2]
            prob *= info['prob'] * p[info['a'], s1, s2]
        probs.append(prob)
        # print('\t>>> {:>2d}, {}'.format(len(path), prob), path)
    probs = list(sorted(probs))
    print(sum(probs))
    # 渲染
    g.render(cleanup=True, format='png')
    return edges, probs


def plot_data(data, probs, prefix):
    fig, axes = plt.subplots(2, 1)
    plt.colorbar(axes[0].imshow(data, cmap='hot'))
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         axes[0].text(j, i, round(data[i, j], 1), size=6,
    #                 ha="center", va="center", color="blue")
    for key, value in probs.items():
        axes[1].plot(value, label=key)
    axes[1].legend()
    plt.savefig('figs/sim_and_prob_{}.png'.format(prefix))


def main():
    num_iter = 1
    size_ = 30
    num_ladders = 0
    num_targets = 1
    num_obstacles = 270

    result_array = []
    for episode in range(num_iter):
        print('{}/{}'.format(episode + 1, num_iter))
        result = run(episode+1, size_, num_ladders, num_targets, num_obstacles)
        result_array.append(result)
    result_array = np.array(result_array)
    print(num_iter, len(result_array), result_array[:, 1:].mean(axis=0))
    print(result_array)


if __name__ == '__main__':
    main()
