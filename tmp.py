import numpy as np
from graphviz import Digraph

from env import SnakeDiscreteEnv
from algo.basic import *
from common.utils import extract_all_paths


def train_and_test(env, algo, max_len=100, **kwargs):
    print('\n------------------------------------------')
    # Train
    env.reset(reuse=True, verbose=True)
    algo = algo(env, **kwargs)
    agent = algo.update()

    # Test
    state = env.reset(reuse=True)
    return_val, step = 0.0, 0
    while True:
        action = agent.action_sample(state)
        next_state, reward, done, terminated = env.step(action, verbose=True)
        return_val += reward
        step += 1
        state = next_state
        env.render(mode=algo.name + ': {}/{}'.format(step, max_len))
        if done or terminated or step >= max_len:
            break
    print('Total reward:', return_val)
    print('Total step:{}/{}'.format(step, max_len))
    print('Final: done:{}), terminated({})'.format(done, terminated))
    print('------------------------------------------')
    return agent, return_val, done


def transition(node_dict, state, policy, env, step, max_len):
    if state in node_dict.keys():
        return

    idx = env.state_list.index(state)
    act_prob = policy[idx]
    acts = np.argwhere(act_prob == act_prob.max())

    pos = env.state_dict[state]['pos']
    coin_checker = env.coin_checker.copy()

    node_dict[state] = {}
    for act in acts.squeeze(axis=1):
        env.pos = pos
        env.coin_checker = coin_checker.copy()
        new_state, _, done, terminated = env.step(act)
        # if new_state == state:
        #     continue
        node_dict[state][new_state] = {
            'a': act, 's_prime': new_state, 'prob': act_prob[act]
        }
        if done or terminated:
            return
        transition(node_dict, new_state, policy, env, step + 1, max_len)
    if step >= max_len:
        return


def plot_tree(node_dict, p, env, filename):
    state_list, state_dict = env.state_list, env.state_dict

    g = Digraph('G', filename='figs/' + filename + '.gv')
    edges = []
    for start, node_info in node_dict.items():
        # name_s = str(start)
        info_s = state_dict[start]
        name_s = '{}({})'.format(info_s['pos'], info_s['status'])
        g.node(name_s, label=name_s)
        for end, v in node_info.items():
            # name_e = str(end)
            info_e = state_dict[end]
            name_e = '{}({})'.format(info_e['pos'], info_e['status'])
            g.edge(name_s, name_e, label='{}({:>4.2f})'.format(v['a'], v['prob']))
            edges.append((start, end))
    # 提取所有最优路径
    all_paths = extract_all_paths(edges)
    probs = []
    for path in all_paths:
        print(path)
        prob = 1.0
        for i, s1 in enumerate(path[:-1]):
            s2 = path[i+1]
            info = node_dict[s1][s2]
            idx1 = state_list.index(s1)
            idx2 = state_list.index(s2)
            prob *= info['prob'] * p[info['a'], idx1, idx2]
        probs.append(prob)
    probs = list(sorted(probs))
    print(sum(probs))
    # 渲染
    g.render(cleanup=True, format='png')
    return edges, probs


def run(episode, **kwargs_env):
    # Environment
    env = SnakeDiscreteEnv(**kwargs_env)
    # Parameters
    kwargs_algo = {
        'gamma': 0.95,
        'max_len': int(1e2),
        'eval_iter': 128,
        'improve_iter': 1000,
    }
    # Algo: PI
    agent, _, done = train_and_test(
        env, algo=PolicyIteration, **kwargs_algo
    )
    # Optimal Paths
    if done:
        node_dict = {}
        state = env.reset(reuse=True)
        transition(node_dict, state, agent.pi, env, 0, kwargs_algo['max_len'])
        plot_tree(node_dict, agent.p, env, filename='graph1_{}'.format(episode))
    env.close()


def main():
    num_iter = 1
    parameters = {
        'size': 30,
        'num_ladder': 0,
        'num_coin': 2,
        'num_mud': 90,
        'num_barrier': 90
    }

    for episode in range(num_iter):
        print('{}/{}'.format(episode + 1, num_iter))
        run(episode+1, **parameters)


if __name__ == '__main__':
    main()
