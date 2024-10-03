import numpy as np
from graphviz import Digraph

from algo.basic import *
from env import SnakeDiscreteEnv
from common.utils import extract_all_paths


def train_and_test(env, algo, max_len=100, **kwargs):
    print('\n------------------------------------------')
    # Train
    env.reset(reuse=True, verbose=False)
    env.ranges = kwargs['prefix']
    algo_instance = algo(env, **kwargs)
    prefix = '_'.join([str(x) for x in kwargs['prefix']])
    agent = algo_instance.update(prefix=prefix)

    # Test
    state = env.reset(reuse=True)
    return_val, step = 0.0, 0
    while True:
        action = agent.action_sample(state)
        next_state, reward, done, terminated = env.step(action, verbose=False)
        return_val += reward
        step += 1
        state = next_state
        env.render(mode=algo_instance.name + ': {}/{}'.format(step, max_len))
        if done or terminated or step >= max_len:
            break
    print('Total reward:', return_val)
    print('Total step:{}/{}'.format(step, max_len))
    print('Final: done({}), terminated({})'.format(done, terminated))

    # Find all optimal paths
    node_dict, edges = {}, []
    state = env.reset(reuse=True)
    transition(node_dict, state, agent.pi, env, 0, max_len)
    prob = plot_tree(
        node_dict, agent.p, env,
        filename='graph_{}'.format(prefix)
    )
    print('Optimal Path: {}'.format(prob))
    print('------------------------------------------')
    return agent, node_dict, edges, return_val, done


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
        info_s = state_dict[start]
        name_s = '{}({})'.format(info_s['pos'], info_s['status'])
        g.node(name_s, label=name_s)
        for end, v in node_info.items():
            info_e = state_dict[end]
            name_e = '{}({})'.format(info_e['pos'], info_e['status'])
            g.edge(name_s, name_e, label='{}({:>4.2f})'.format(v['a'], v['prob']))
            edges.append((start, end))
    g.render(cleanup=True, format='png')  # 渲染

    all_paths = extract_all_paths(edges)  # 提取所有最优路径
    probs = []
    for path in all_paths:
        prob = 1.0
        for i, s1 in enumerate(path[:-1]):
            s2 = path[i + 1]
            info = node_dict[s1][s2]
            idx1 = state_list.index(s1)
            idx2 = state_list.index(s2)
            prob *= info['prob'] * p[info['a'], idx1, idx2]
        probs.append(prob)
    return sum(list(sorted(probs)))


def run(episode, **kwargs_env):
    # Environment
    env = SnakeDiscreteEnv(**kwargs_env)
    # Parameters
    kwargs_algo = {
        'gamma': 0.95,
        'max_len': int(1e2),
        'eval_iter': 128,
        'improve_iter': 1000,
        'prefix': None
    }
    # Algo: PI
    # agent, node_dict, edges, *_ = train_and_test(env, algo=PolicyIteration, **kwargs_algo)
    h = kwargs_algo['max_len'] + 50
    for i in range(h+1, h+2):
        for j in range(-1, 0):
            for k in range(h+1, h+2):
                print('\n(x,y,z): ({},{},{})'.format(i, j, k))
                kwargs_algo['prefix'] = (k, j, i, 1)
                result = train_and_test(env, algo=PolicyIteration, **kwargs_algo)
                if result[-1]:
                    break
    env.close()


def main():
    num_iter = 1
    parameters = {
        'size': 4,
        'num_ladder': 0,
        'num_coin': 1,
        'num_mud': 2,
        'num_barrier': 2
    }

    for episode in range(num_iter):
        print('{}/{}'.format(episode + 1, num_iter))
        run(episode + 1, **parameters)


if __name__ == '__main__':
    main()
