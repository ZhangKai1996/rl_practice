import numpy as np

from env import SnakeEnv
from algo import PolicyIteration

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
        env.render()
        return_val += reward
        step += 1
        state = next_state

        if step >= max_len:
            break

    print('total reward:', return_val)
    print('total step:', step)


def train():
    env = SnakeEnv(ladder_num=ladder_num, size=size)
    print('-------------Policy Iteration-------------')
    trainer_pi = PolicyIteration(env)
    policy_pi = trainer_pi.update()
    test(env, policy_pi)
    env.close()


if __name__ == '__main__':
    train()
