import numpy as np
from graphviz import Digraph

from algo.basic import *
from common.utils import extract_all_paths

from env import SnakeDiscreteEnv


def train(env, algo, **kwargs):
    print('-------------------Training-----------------')
    env.reset()
    algo_instance = algo(env, **kwargs)
    return algo_instance.update(prefix=env.prefix)


def test(env, agent, max_len=100, reuse=True, **kwargs):
    print('-------------------Testing------------------')
    state = env.reset(reuse=reuse)

    return_val = 0.0
    step = 0
    path = [state]
    while True:
        action = agent.action_sample(state)
        next_state, reward, done, terminated = env.step(action, verbose=True)
        state = next_state
        env.render(
            mode='Algo: {}, '.format(kwargs['name']) +
                 'Step: {}/{}, '.format(step, max_len) +
                 'Parameters: {}'.format(env.prefix)
        )

        path.append(next_state)
        return_val += reward
        step += 1
        if done or terminated or step >= max_len:
            break

    print('Path: ', path)
    print('Total reward:', return_val)
    print('Total step: {}/{}'.format(step, max_len))
    print('Final: done({}), terminated({})'.format(done, terminated))
    return [return_val, step, done]


def extract_path(env, agent, max_len):
    state = env.reset(reuse=True)

    print('Collecting all paths ...')
    node_dict, edges = {}, []
    random_path(node_dict, state, env, 0, max_len)

    print('Plotting tree ...')
    seqs, paths = plot_tree(
        env=env,
        node_dict=node_dict,
        filename='graph_{}'.format(env.prefix)
    )

    print('Feasible Path: {}'.format(len(paths)))
    for i, seq in enumerate(paths):
        prob = 1.0
        for s, a, s_prime in seq:
            prob *= agent.pi[s, a] * agent.p[a, s, s_prime]
        print('\tPath {:>3d}: {:>5.3f}'.format(i+1, prob), seqs[i])
    return node_dict, edges


def random_path(node_dict, state, env, step, max_len):
    if state in node_dict.keys():
        return
    if step >= max_len:
        return

    key = '{}_{}'.format(state, step)
    node_dict[key] = {}
    iterator = list(range(env.action_space.n))
    np.random.shuffle(iterator)
    for act in iterator:
        env.pos = state
        new_state, _, done, terminated = env.step(act)
        node_dict[key][act] = {
            's_prime': '{}_{}'.format(new_state, step+1),
            'prob': 0.0 if terminated or step + 1 >= max_len else 0.25
        }
        if done or terminated:
            continue
        random_path(node_dict, new_state, env, step + 1, max_len)


def plot_tree(env, node_dict, filename):
    g = Digraph('G', filename='figs/' + filename + '.gv')

    edges = []
    dict1 = {}
    for name_s, node_info in node_dict.items():
        g.node(name_s, label=name_s)
        for act, v in node_info.items():
            name_e = v['s_prime']
            if name_s in dict1.keys():
                dict1[name_s][name_e] = act
            else:
                dict1[name_s] = {name_e: act}
            if v['prob'] == 0.0:
                continue
            g.edge(name_s, name_e, label='{}({:>4.2f})'.format(act, v['prob']))
            edges.append((name_s, name_e))
    g.render(cleanup=True, format='png')  # 渲染
    paths, seqs = [], []
    for path in extract_all_paths(edges):
        seq = [int(x[0]) for x in path]
        path_ = [(int(x[0]), dict1[x][path[i+1]], int(path[i+1][0]))
                 for i, x in enumerate(path[:-1])]
        if seq[-1] in env.coins and len(seq) == len(set(seq)):
            seqs.append(seq)
            paths.append(path_)
    return seqs, paths


def run(**kwargs_env):
    # Environment
    env = SnakeDiscreteEnv(**kwargs_env)
    # Parameters
    kwargs_algo = {
        'name': 'PI',
        'gamma': 0.95,
        'max_len': 40,
        'eval_iter': 128,
        'improve_iter': 1000,
    }

    agent = train(env, algo=PolicyIteration, **kwargs_algo)
    result1 = test(env, agent=agent, **kwargs_algo)
    result2 = test(env, agent=agent, reuse=False, **kwargs_algo)
    print('------------------------------------------')
    env.close()


if __name__ == '__main__':
    num_episode = 1
    env_params = {
        'size': 20,
        'num_coin': 3,
        'num_land': 0,
        'num_mud': 20,
        'num_barrier': 20,
        'rew_setting': (-10, -2, 0, 10, 10),
    }
    for episode in range(num_episode):
        print('{}/{}'.format(episode + 1, num_episode))
        run(**env_params)
