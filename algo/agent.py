import copy

import cv2
import numpy as np


class PlanningAgent:
    def __init__(self, env):
        self.num_obs = num_obs = env.observation_space.n
        self.num_act = num_act = env.action_space.n

        self.p = np.zeros([num_act, num_obs, num_obs], dtype=np.float32)  # P(s'|s,a)
        for a in range(num_act):
            for s in range(num_obs):
                s_prime = env.execute_action(a, s)
                if s_prime not in env.ladders.keys():
                    self.p[a, s, s_prime] = 1.0
                    continue
                for s_prime, prob in zip(*env.ladders[s_prime]):
                    self.p[a, s, s_prime] = prob
        self.r = [env.get_reward(s)[0] for s in range(num_obs)]     # R(s')
        random_actions = np.random.randint(0, num_act, size=(num_obs, ))
        self.pi = np.eye(num_act)[random_actions]                   # $\pi$(s)
        self.v = np.zeros((num_obs,))                               # V(s)
        self.q = np.zeros((num_obs, num_act))                       # Q(s,a)

        self.env = env
        self.render = None

    def visual(self, algo):
        if self.render is None:
            self.render = ValueRender(env=self.env)
        self.render.draw(
            # values={'v': self.v, 'q': self.q},
            values={'v': self.v, 'r': self.r},
            algo=algo
        )


class LearningAgent:
    def __init__(self, env):
        self.num_obs = num_obs = env.observation_space.n
        self.num_act = num_act = env.action_space.n

        self.r = [env.get_reward(s)[0] for s in range(num_obs)]     # R(s')
        random_actions = np.random.randint(0, num_act, size=(num_obs, ))
        self.pi = np.eye(num_act)[random_actions]                # $\pi$(s)
        self.q = np.zeros((num_obs, num_act))
        self.n = np.zeros((num_obs, num_act))
        self.track_q = np.zeros((num_obs, num_act))

        self.env = env
        self.render = None

    def play(self, s, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_act)
        return np.argmax(self.pi[s])

    def visual(self, algo):
        if self.render is None:
            self.render = ValueRender(env=self.env)

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
            # values={'v': v, 'q': self.q},
            values={'v': v, 'r': self.r},
            algo=algo
        )


class ValueRender:
    def __init__(self, env, width=1200, height=1200, padding=0.05):
        self.width = width
        self.height = height
        self.padding = padding
        self.env = env
        self.base_img = np.ones((height, width, 3), np.uint8) * 255

    def draw(self, algo: str, values: dict):
        w_p = int(self.width * self.padding)
        h_p = int(self.height * self.padding)
        size = self.env.size
        border_len = int((self.width - w_p * 2) / size)

        images = []
        if 'v' in values.keys():
            base_img = self.__draw_v(values['v'], w_p, h_p, size, border_len)
            images.append(base_img)
        if 'q' in values.keys():
            base_img = self.__draw_q(values['q'], w_p, h_p, size, border_len)
            images.append(base_img)
        if 'r' in values.keys():
            base_img = self.__draw_v(values['r'], w_p, h_p, size, border_len)
            images.append(base_img)
        if len(images) > 0:
            # cv2.imwrite('figs/value_{}.png'.format(algo), np.hstack(images))
            import time
            cv2.imwrite('figs/value_{}_{}.png'.format(algo, int(time.time())), np.hstack(images))

    def __draw_a_piece(self, base_img, poses, is_max=0):
        # 画三角形
        cv2.polylines(base_img, [poses], True, (0, 0, 0))
        # 画红线
        if is_max != 0:
            p1, p2 = poses[0], poses[2]
            cv2.line(
                base_img,
                poses[1], (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)),
                (0, 0, 255)
            )
        return base_img

    def __draw_q(self, value, w_p, h_p, size, border_len):
        base_img = copy.deepcopy(self.base_img)

        for i in range(size):
            column = int(border_len / 2 + i * border_len + w_p)
            # Draw state
            cv2.putText(
                base_img, str(i), (column, int(h_p/2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

            for j in range(size):
                row = int(border_len / 2 + j * border_len + h_p)
                pos = (column, row)
                idx = i + j * size

                x1_out = int(pos[0] - border_len / 2)
                x2_out = int(pos[0] + border_len / 2)
                y1_out = int(pos[1] - border_len / 2)
                y2_out = int(pos[1] + border_len / 2)

                # Draw targets
                if idx in self.env.targets:
                    cv2.circle(base_img, pos, 10, (0, 255, 0), thickness=-1)
                if idx == self.env.start:
                    cv2.circle(base_img, pos, 10, (255, 0, 0), thickness=2)

                q_act = value[idx]
                is_max = [0 for _ in q_act]
                argmax_idx = np.argmax(q_act)
                is_max[argmax_idx] = 1
                # 0-up
                poses = np.array([(x1_out, y1_out), pos, (x2_out, y1_out)], np.int32)
                base_img = self.__draw_a_piece(base_img, poses, is_max=is_max[0])
                # 1-down
                poses = np.array([(x1_out, y2_out), pos, (x2_out, y2_out)], np.int32)
                base_img = self.__draw_a_piece(base_img, poses, is_max=is_max[1])
                # 2-right
                poses = np.array([(x2_out, y1_out), pos, (x2_out, y2_out)], np.int32)
                base_img = self.__draw_a_piece(base_img, poses, is_max=is_max[2])
                # 3-left
                poses = np.array([(x1_out, y1_out), pos, (x1_out, y2_out)], np.int32)
                base_img = self.__draw_a_piece(base_img, poses, is_max=is_max[3])

                # Draw state
                if i == 0:
                    cv2.putText(
                        base_img, str(j), (int(h_p / 2), row),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )

        return base_img

    def __draw_v(self, value, w_p, h_p, size, border_len):
        base_img = copy.deepcopy(self.base_img)

        value_pi = value - np.min(value)
        max_v_pi = np.max(value_pi)

        for i in range(size):
            column = int(border_len / 2 + i * border_len + w_p)
            # Draw state
            cv2.putText(
                base_img, str(i), (column, int(h_p/2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

            for j in range(size):
                row = int(border_len / 2 + j * border_len + h_p)
                pos = (column, row)
                v_pi = value_pi[i + j * size]
                if max_v_pi > 0:
                    pixel = int((1 - v_pi / max_v_pi) * 255)
                else:
                    pixel = 0
                # Draw grids
                cv2.rectangle(
                    base_img,
                    (int(pos[0] - border_len / 2), int(pos[1] - border_len / 2)),
                    (int(pos[0] + border_len / 2), int(pos[1] + border_len / 2)),
                    (pixel, pixel, pixel),
                    thickness=-1
                )
                cv2.putText(
                    base_img, str(round(v_pi/max_v_pi, 2)), (pos[0] - 10, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), 1, cv2.LINE_AA
                )
                # Draw state
                if i == 0:
                    cv2.putText(
                        base_img, str(j), (int(h_p / 2), row),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )
        return base_img

    def close(self):
        cv2.waitKey(1) & 0xFF
        cv2.destroyAllWindows()
