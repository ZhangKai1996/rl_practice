import numpy as np
import tensorflow.compat.v1 as tf

from common import ValueRender

tf.compat.v1.disable_eager_execution()


class PlanningAgent:
    def __init__(self, env, **kwargs):
        self.env = env
        self.num_obs = env.observation_space.n
        self.num_act = env.action_space.n
        self.__build_model()
        self.render = None

    def __build_model(self):
        env = self.env
        num_obs, num_act = self.num_obs, self.num_act

        self.p = np.zeros([num_act, num_obs, num_obs], dtype=np.float32)  # P(s'|s,a)
        self.r = np.zeros([num_act, num_obs, num_obs], dtype=np.float32)  # R(s')
        self.r_ = np.zeros([num_obs, num_act], dtype=np.float32)
        for s in range(num_obs):
            if s in env.barriers: continue
            for a in range(num_act):
                env.pos = s
                s_prime, reward, *_ = env.step(a)
                self.r[a, s, s_prime] = reward
                self.r_[s, a] = reward
                self.p[a, s, s_prime] = 1.0
        random_actions = np.random.randint(0, num_act, size=(num_obs,))
        self.pi = np.eye(num_act)[random_actions]    # pi(s)
        self.v = np.zeros((num_obs,))                # V(s)
        self.q = np.zeros((num_obs, num_act))        # Q(s,a)

    def action_sample(self, state):
        act_prob = self.pi[state]
        acts = np.argwhere(act_prob == act_prob.max())
        acts = acts.squeeze(axis=1)
        np.random.shuffle(acts)
        return acts[0]

    def visual(self, algo):
        if self.render is None:
            self.render = ValueRender(env=self.env)
        r_state = np.zeros((self.num_obs,))                # V(s)
        for s_prime in range(self.num_obs):
            for s in range(self.num_obs):
                for a in range(self.num_act):
                    r_state[s_prime] += self.pi[s, a] * self.p[a, s, s_prime] * self.r[a, s, s_prime]
        self.render.draw(
            values={'v': self.v,
                    'r': r_state,
                    'q': self.r_,
                    'pi': self.pi},
            algo=algo
        )


class LearningAgent:
    def __init__(self, env, **kwargs):
        self.env = env
        self.num_obs = env.observation_space.n
        self.num_act = env.action_space.n
        self.__build_model()
        self.render = None

    def __build_model(self):
        env = self.env
        num_obs, num_act = self.num_obs, self.num_act

        self.p = None
        self.r = np.zeros([num_act, num_obs, num_obs], dtype=np.float32)                  # R(s')
        for s in range(num_obs):
            if s in env.barriers: continue
            for a in range(num_act):
                env.pos = s
                s_prime, reward, *_ = env.step(a)
                self.r[a, s, s_prime] = reward
        random_actions = np.random.randint(0, num_act, size=(num_obs,))
        self.pi = np.eye(num_act)[random_actions]    # pi(s)
        self.q = np.zeros((num_obs, num_act))        # Q(s,a)
        self.n = np.zeros((num_obs, num_act))

    def action_sample(self, state, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_act)
        idx = self.env.state_list.index(state)
        act_prob = self.pi[idx]
        acts = np.argwhere(act_prob == act_prob.max())
        acts = acts.squeeze(axis=1)
        np.random.shuffle(acts)
        return acts[0]

    def visual(self, algo):
        if self.render is None:
            self.render = ValueRender(env=self.env, algo=algo)
        v = np.zeros((self.num_obs,))  # 根据策略和状态动作值函数计算值函数
        for s in range(self.num_obs):
            p_act = self.pi[s]
            value = 0.0
            for a in range(self.num_act):
                prob = p_act[a]
                value += prob * self.q[s, a]
            v[s] = value
        self.render.draw(values={'v': v, 'r': self.r}, algo=algo)  # 画出V值表和Q值表


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
