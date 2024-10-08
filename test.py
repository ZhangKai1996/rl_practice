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
    path, refresh = [], True
    while True:
        action = agent.action_sample(state)
        next_state, reward, done, terminated = env.step(action, verbose=False)
        path.append(next_state)
        return_val += reward
        step += 1
        state = next_state
        env.render(mode= 'Algo: {}, '.format(algo_instance.name) +
                         'Step: {}/{}, '.format(step, max_len) +
                         'Parameters: {}'.format(prefix),
                   refresh=refresh)
        refresh = False
        if done or terminated or step >= max_len:
            break
    is_optimal = (done and
                  len(path[:-1]) == len(env.land) and
                  all([x == y for x, y in zip(env.land, path[:-1])]))
    # is_optimal = all([point in path for point in env.land])
    print('Path: ', path)
    print('Targets: ', env.land)
    print('Total reward:', return_val)
    print('Total step:{}/{}'.format(step, max_len))
    print('Final: done({}), optimal({}), terminated({})'.format(
        done, is_optimal, terminated))
    # Find all optimal paths
    node_dict, edges = {}, []
    # if done:
    algo_instance.render(prefix=prefix)
    state = env.reset(reuse=True)
    transition(node_dict, state, agent.pi, env, 0, max_len)
    prob = plot_tree(node_dict, agent.p, filename='graph_{}'.format(prefix))
    print('Optimal Path: {}'.format(prob))
    print('------------------------------------------')
    return agent, node_dict, edges, [return_val, step, done, is_optimal]


def transition(node_dict, state, policy, env, step, max_len):
    if state in node_dict.keys():
        return

    act_prob = policy[state]
    node_dict[state] = {}
    for act in range(env.action_space.n):
        env.pos = state
        new_state, _, done, terminated = env.step(act)
        prob = act_prob[act]
        is_max = prob == max(act_prob)
        node_dict[state][act] = {
            's_prime': new_state,
            'prob': prob,
            'is_max': is_max,
            'step': step
        }
        if done or terminated:
            continue
        transition(node_dict, new_state, policy, env, step + 1, max_len)
    if step >= max_len:
        return


def plot_tree(node_dict, p, filename):
    g = Digraph('G', filename='figs/' + filename + '.gv')
    edges = []
    for start, node_info in node_dict.items():
        for act, v in node_info.items():
            end = v['s_prime']
            name_s = '{}({})'.format(start, v['step'])
            name_e = '{}({})'.format(end, v['step']+1)

            g.node(name_s, label=name_s)
            g.edge(
                name_s, name_e,
                label='{}({:>4.2f})'.format(act, v['prob']),
                color='green' if v['is_max'] else 'gray'
            )
            edges.append((start, end))
    g.render(cleanup=True, format='png')  # 渲染

    # all_paths = extract_all_paths(edges)  # 提取所有最优路径
    # probs = []
    # for path in all_paths:
    #     prob = 1.0
    #     for i, s1 in enumerate(path[:-1]):
    #         s2 = path[i + 1]
    #         info = node_dict[s1][s2]
    #         prob *= info['prob'] * p[info['a'], s1, s2]
    #     probs.append(prob)
    probs = [0.0, ]
    return sum(list(sorted(probs)))


def run(episode, **kwargs_env):
    # Environment
    env = SnakeDiscreteEnv(**kwargs_env)
    # Parameters
    kwargs_algo = {
        'gamma': 0.95,
        'max_len': 30,
        'eval_iter': 128,
        'improve_iter': 1000,
        'prefix': None
    }
    # Algo: PI
    # agent, node_dict, edges, *_ = train_and_test(env, algo=PolicyIteration, **kwargs_algo)
    h = kwargs_algo['max_len']
    metric = []
    for r_b in np.linspace(-h, -h, 1):
        for r_m in np.linspace(-2, -2, 1):
            for r_e in range(-1, 2):
                for r_l in np.linspace(0, h, int(h/3)+1):
                    for r_t in range(h, h+1):
                        kwargs_algo['prefix'] = (r_b,r_m,r_e,r_l,r_t)
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
        'size': 5,
        'num_ladder': 0,
        'num_coin': 1,
        'num_land': 5,
        'num_mud': 1,
        'num_barrier': 4
    }

    for episode in range(num_iter):
        print('{}/{}'.format(episode + 1, num_iter))
        run(episode + 1, **parameters)


if __name__ == '__main__':
    main()
