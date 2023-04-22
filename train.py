import numpy as np

from env import SnakeEnv
from algo import PolicyIteration, ValueIteration, MonteCarlo

scale = 1
ladder_num = 10 * scale
size = 10 * scale
max_len = 30


def test(env, policy, reuse=True):
    print([np.argmax(p) for p in policy])

    state, done = env.reset(reuse=reuse), False
    return_val = 0
    step = 0

    while not done:
        act = np.argmax(policy[state])
        next_state, reward, done, _ = env.step(act)
        # print('step:{:>3d},state{:>3d},action:{},next state:{:>3d},reward:{:>+4d},done:{}'.format(
        #     step, state, act, next_state, reward, int(done))
        # )
        env.render()
        return_val += reward
        step += 1
        state = next_state

        if step >= max_len:
            break

    print('total reward:', return_val)
    print('total step:', step)


def train_pi_and_vi():
    env = SnakeEnv(ladder_num=ladder_num, size=size)
    print('-------------Policy Iteration-------------')
    trainer_pi = PolicyIteration(env, eval_max_iter=10)
    policy_pi = trainer_pi.update()
    test(env, policy_pi)

    print('-------------Value Iteration-------------')
    env.reset(reuse=True)
    trainer_vi = ValueIteration(env, eval_max_iter=10)
    for _ in range(10):
        policy_vi = trainer_vi.update()
        test(env, policy_vi)

    env.close()


def train_mc():
    env = SnakeEnv(ladder_num=ladder_num, size=size)
    print('-------------Policy Iteration-------------')
    trainer_pi = PolicyIteration(env, eval_max_iter=10)
    policy_pi = trainer_pi.update()
    test(env, policy_pi)

    print('---------------Monte Carlo---------------')
    env.reset(reuse=True)
    trainer_mc = MonteCarlo(env, epsilon=0.5, max_len=max_len)
    policy_mc = trainer_mc.update()
    test(env, policy_mc)

    env.close()


if __name__ == '__main__':
    # train_pi_and_vi()
    train_mc()
