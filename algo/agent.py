import numpy as np
import tensorflow.compat.v1 as tf

from common import ValueRender
tf.compat.v1.disable_eager_execution()


class PlanningAgent:
    def __init__(self, env, rew=0, **kwargs):
        self.num_obs = num_obs = env.observation_space.n
        self.num_act = num_act = env.action_space.n

        self.p = np.zeros([num_act, num_obs, num_obs], dtype=np.float32)  # P(s'|s,a)
        for s in range(num_obs):
            if s in env.obstacles: continue
            for a in range(num_act):
                s_prime = env.execute_action(a, s)
                if s_prime not in env.ladders.keys():
                    self.p[a, s, s_prime] = 1.0
                    continue
                for s_prime, prob in zip(*env.ladders[s_prime]):
                    self.p[a, s, s_prime] = prob
        if rew == 0:
            self.r = np.array([env.get_reward(s)[0] for s in range(num_obs)])     # R1(s')
        elif rew == 1:
            self.r = np.array([env.get_reward1(s)[0] for s in range(num_obs)])    # R2(s')
        elif rew == 2:
            self.r = np.array([env.get_reward(s, scale=10.0)[0] for s in range(num_obs)])    # R2(s')
        elif rew == 3:
            self.r = np.array([env.get_reward1(s, scale=10.0)[0] for s in range(num_obs)])  # R2(s')
        else:
            self.r = np.array([np.random.random() for _ in range(num_obs)])

        random_actions = np.random.randint(0, num_act, size=(num_obs,))
        self.pi = np.eye(num_act)[random_actions]                       # $\pi$(s)
        self.v = np.zeros((num_obs,))                                   # V(s)
        self.q = np.zeros((num_obs, num_act))                           # Q(s,a)

        self.env = env
        self.render = None

    def action_sample(self, state):
        # return np.argmax(self.agent.pi[state])
        act_prob = self.pi[state]
        acts = np.argwhere(act_prob == act_prob.max())
        acts = acts.squeeze(axis=1)
        np.random.shuffle(acts)
        return acts[0]

    def visual(self, algo):
        if self.render is None:
            self.render = ValueRender(env=self.env)
        self.render.draw(
            # values={'v': self.v, 'q': self.q},
            values={'v': self.v, 'r': self.r},
            algo=algo
        )


class LearningAgent:
    def __init__(self, env, rew=0):
        self.num_obs = num_obs = env.observation_space.n
        self.num_act = num_act = env.action_space.n

        if rew == 0:
            self.r = np.array([env.get_reward(s)[0] for s in range(num_obs)])     # R1(s')
        elif rew == 1:
            self.r = np.array([env.get_reward1(s)[0] for s in range(num_obs)])    # R2(s')
        elif rew == 2:
            self.r = np.array([env.get_reward(s, scale=10.0)[0] for s in range(num_obs)])    # R2(s')
        elif rew == 3:
            self.r = np.array([env.get_reward1(s, scale=10.0)[0] for s in range(num_obs)])  # R2(s')
        else:
            self.r = np.array([np.random.random() for _ in range(num_obs)])
        # self.r = [env.get_reward(s)[0] for s in range(num_obs)]  # R(s')
        random_actions = np.random.randint(0, num_act, size=(num_obs,))
        self.pi = np.eye(num_act)[random_actions]           # $\pi$(s)
        self.q = np.zeros((num_obs, num_act))
        self.n = np.zeros((num_obs, num_act))

        self.env = env
        self.render = None

    def play(self, s, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_act)
        # return np.argmax(self.pi[s])

        act_prob = self.pi[s]
        acts = np.argwhere(act_prob == act_prob.max())
        acts = acts.squeeze(axis=1)
        np.random.shuffle(acts)
        return acts[0]

    def visual(self, algo):
        if self.render is None:
            self.render = ValueRender(env=self.env, algo=algo)

        # 根据策略和状态动作值函数计算值函数
        v = np.zeros((self.num_obs,))
        for s in range(self.num_obs):
            p_act = self.pi[s]
            value = 0.0
            for a in range(self.num_act):
                prob = p_act[a]
                value += prob * self.q[s, a]
            v[s] = value

        # 画出V值表和Q值表
        self.render.draw(
            values={'v': v, 'q': self.q},
            # values={'v': v, 'r': self.r},
        )


class NetworkAgent:
    def __init__(self, env, lr=0.005):
        self.dim_obs = list(env.observation_space.shape)
        self.num_act = env.action_space.n

        self.obs_ph = tf.placeholder(tf.float32, shape=[None, ] + self.dim_obs)
        self.act_ph = tf.placeholder(tf.float32, shape=[None, self.num_act])
        self.rew_ph = tf.placeholder(tf.float32, shape=[None, ])

        self.logit = self.__build_net(self.obs_ph, hidden_size=64)
        self.policy = tf.nn.softmax(self.logit)
        self.log_policy = tf.nn.log_softmax(self.logit)
        self.log_prob = tf.reduce_sum(self.act_ph * self.log_policy, axis=1)
        self.loss = - tf.reduce_mean(self.log_prob * self.rew_ph)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # self.params = tf.trainable_variables()
        # self.gradients = optimizer.compute_gradients(self.loss, var_list=self.params)
        # self.train_op = optimizer.apply_gradients(self.gradients)
        self.train_op = optimizer.minimize(self.loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def __build_net(self, inputs, hidden_size=64, name='pi'):
        with tf.name_scope(name):
            out = inputs
            out = tf.layers.dense(out, units=hidden_size, activation=tf.nn.relu)
            out = tf.layers.dense(out, units=self.num_act)
        return out

    def play(self, s, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_act)
        policy = self.session.run(self.policy, {self.obs_ph: s[np.newaxis]})[0]
        return np.argmax(policy)

    def learn(self, obs_batch, act_batch, rew_batch, done_batch):
        feed = {
            self.obs_ph: obs_batch,
            self.act_ph: np.eye(self.num_act)[act_batch.astype(int)],
            self.rew_ph: rew_batch
        }
        [loss, _] = self.session.run([self.loss, self.train_op], feed)
        return loss
