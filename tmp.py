import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt

from algo.basic import *
from algo.value_based import *
from env import SnakeDiscreteEnv
from common.utils import extract_all_paths


def train_and_test(env, algo, max_len=100, **kwargs):
    print('\n------------------------------------------')
    # Train
    env.reset(reuse=True, verbose=False)
    env.ranges = kwargs['prefix']
    algo_instance = algo(env, **kwargs)
    prefix = '_'.join([str(x) for x in kwargs['prefix']])
    print('Reward: ', prefix)
    agent = algo_instance.update()

    # Test
    state = env.reset(reuse=True)
    return_val, step = 0.0, 0
    path, refresh = [state], True
    while True:
        action = agent.action_sample(state)
        next_state, reward, done, terminated = env.step(action, verbose=False)
        path.append(next_state)
        return_val += reward
        step += 1
        state = next_state
        env.render(mode='Algo: {}, '.format(algo_instance.name) +
                        'Step: {}/{}, '.format(step, max_len) +
                        'Parameters: {}'.format(prefix),
                   refresh=refresh)
        refresh = False
        if done or terminated or step >= max_len:
            break
    is_optimal = (done and
                  len(path[1:-1]) == len(env.land) and
                  all([x == y for x, y in zip(env.land, path[1:-1])]))
    # is_optimal = all([point in path for point in env.land])
    print('Path: ', path)
    print('Planned points: ', env.land)
    print('Total reward:', return_val)
    print('Total step:{}/{}'.format(step, max_len))
    print('Final: done({}), optimal({}), terminated({})'.format(
        done, is_optimal, terminated))
    algo_instance.render(prefix=prefix)
    # Find all optimal paths
    node_dict, edges = {}, []
    if kwargs['plot_tree']:
        state = env.reset(reuse=True)
        random_path(node_dict, state, env, 0, max_len)
        paths = plot_tree(node_dict, filename='graph_{}'.format(prefix))
        print('Feasible Path: {}'.format(len(paths)))
        for i, path in enumerate(paths):
            print('\t', i+1, path)
        idx = np.random.randint(0, len(paths))
        env.land = paths[idx][1:-1]
    print('------------------------------------------')
    return agent, node_dict, edges, [return_val, step, done, is_optimal]


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
        if new_state == state or terminated:
            continue
        prob = 0.0 if step + 1 >= max_len else 0.25
        node_dict[key][act] = {'s_prime': '{}_{}'.format(new_state, step+1),
                               'prob': prob}
        if done:
            continue
        random_path(node_dict, new_state, env, step + 1, max_len)


def plot_tree(node_dict, filename):
    g = Digraph('G', filename='figs/' + filename + '.gv')
    edges = []
    for name_s, node_info in node_dict.items():
        g.node(name_s, label=name_s)
        for act, v in node_info.items():
            if v['prob'] == 0.0:
                continue
            name_e = v['s_prime']
            g.edge(name_s, name_e, label='{}({:>4.2f})'.format(act, v['prob']))
            edges.append((name_s, name_e))
    g.render(cleanup=True, format='png')  # 渲染
    all_paths = []
    for path in extract_all_paths(edges):
        all_paths.append([int(x[0]) for x in path])
    return [path for path in all_paths
            if len(set(path)) == len(path)]


def run(episode, **kwargs_env):
    # Environment
    env = SnakeDiscreteEnv(**kwargs_env)
    # Parameters
    kwargs_algo = {
        'gamma': 0.95,
        'max_len': 10,
        'eval_iter': 128,
        'improve_iter': 1000,
        'prefix': (-10, -2, 0, 10, 10),
        'plot_tree': True
    }
    train_and_test(env, algo=PolicyIteration, **kwargs_algo)
    kwargs_algo['plot_tree'] = False
    # Algo: PI
    # agent, node_dict, edges, *_ = train_and_test(env, algo=PolicyIteration, **kwargs_algo)
    h = kwargs_algo['max_len']
    metric = []
    for r_b in np.linspace(-h, -h, 1):
        for r_m in np.linspace(-2, -2, 1):
            for r_e in range(-1, 2):
                for r_l in np.linspace(0, h, int(h / 2) + 1):
                    for r_t in range(h, h + 1):
                        kwargs_algo['prefix'] = (r_b, r_m, r_e, r_l, r_t)
                        result = train_and_test(env, algo=PolicyIteration, **kwargs_algo)
                        # result = train_and_test(env, algo=MonteCarlo, **kwargs_algo)
                        metric.append([r_l, r_e, ] + [float(x) for x in result[-1]])

    metric = np.array(metric)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(metric[:, 0], metric[:, 1], metric[:, 4], marker='o', label='Done')
    ax.scatter(metric[:, 0], metric[:, 1], metric[:, 5], marker='*', label='Optimal')
    ax.set_xlabel('land')
    ax.set_ylabel('empty')
    ax.set_zlabel('0/1')
    ax.legend()
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(metric[:, 0], metric[:, 1], metric[:, 3], marker='o')
    ax.set_xlabel('land')
    ax.set_ylabel('empty')
    ax.set_zlabel('step')
    plt.show()

    env.close()


def main():
    num_iter = 1
    parameters = {
        'size': 3,
        'num_ladder': 0,
        'num_coin': 1,
        'num_land': 0,
        'num_mud': 0,
        'num_barrier': 1
    }

    for episode in range(num_iter):
        print('{}/{}'.format(episode + 1, num_iter))
        run(episode + 1, **parameters)


if __name__ == '__main__':
    main()
