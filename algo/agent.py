import copy
import cv2
import numpy as np


class PlanningAgent(object):
    def __init__(self, env, algo='algo'):
        self.s_len = s_len = env.observation_space.n
        self.a_len = a_len = env.action_space.n

        ladders = env.ladders
        p = np.zeros([a_len, s_len, s_len], dtype=np.float32)
        for a in range(a_len):
            for s in range(s_len):
                s_prime = env.execute_action(a, s)
                if s_prime not in ladders.keys():
                    p[a, s, s_prime] = 1.0
                else:
                    for v1, v2 in zip(*ladders[s_prime]):
                        p[a, s, v1] = v2
        self.p = p  # T(s'|s,a)
        self.r = [env.get_reward(s)[0] for s in range(s_len)]  # R(s)
        random_actions = np.random.randint(0, a_len, size=(s_len, ))
        self.pi = np.eye(a_len)[random_actions]                # $\pi$(s)

        self.state_value = np.zeros((s_len,))  # V(s)
        self.state_action_value = np.zeros((s_len, a_len))  # Q(s,a)

        self.env = env
        self.algo = algo
        self.render = None

    def visual(self, mode='v'):
        if self.render is None:
            self.render = ValueRender(self.env, self.algo, mode=mode)

        if mode == 'v':
            self.render.draw(self.state_value)
        elif mode == 'q':
            self.render.draw(self.state_action_value)
        else:
            raise NotImplementedError


class LearningAgent(object):
    def __init__(self, env, algo='algo'):
        self.s_len = s_len = env.observation_space.n
        self.a_len = a_len = env.action_space.n

        self.pi = np.ones((s_len, a_len)) * 1.0 / a_len
        self.value_q = np.zeros((s_len, a_len))
        self.value_n = np.zeros((s_len, a_len))
        self.track_q = np.zeros((s_len, a_len))

        self.env = env
        self.algo = algo
        self.render = None

    def play(self, s, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.a_len)
        else:
            return np.argmax(self.pi[s])

    def visual(self, mode='q'):
        if self.render is None:
            self.render = ValueRender(self.env, self.algo, mode=mode)

        if mode == 'v':
            pass
            # print("Drawing state action value is so complex!")
        elif mode == 'q':
            self.render.draw(self.value_q)
        else:
            raise NotImplementedError


class ValueRender:
    def __init__(self, env, algo='algo', mode='v', width=1200, height=1200, padding=0.05):
        self.video = cv2.VideoWriter(
            'figs/value_{}_{}.avi'.format(mode, algo),
            cv2.VideoWriter_fourcc(*'MJPG'),
            8,
            (width, height)
        )

        self.width = width
        self.height = height
        self.padding = padding
        self.env = env
        self.mode = mode

        self.base_img = np.ones((height, width, 3), np.uint8) * 255

    def draw(self, value, show=False):
        # 创建一个的白色画布，RGB(255,255,255)为白色
        base_img = copy.deepcopy(self.base_img)

        w_p = int(self.width * self.padding)
        h_p = int(self.height * self.padding)
        size = self.env.size
        border_len = int((self.width - w_p * 2) / size)
        if self.mode == 'v':
            base_img = self.__draw_v(base_img, value, w_p, h_p, size, border_len)
        elif self.mode == 'q':
            base_img = self.__draw_q(base_img, value, w_p, h_p, size, border_len)
        else:
            raise NotImplementedError

        self.video.write(base_img)

        if show:
            cv2.imshow('figs/base image', base_img)
            if cv2.waitKey(0) == 113:
                cv2.destroyAllWindows()

    def __draw_q(self, base_img, value, w_p, h_p, size, border_len):
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
                q_act = value[i + j * size]
                q_act_ = q_act - np.min(q_act)
                max_q = np.max(q_act_)

                x1_out = int(pos[0] - border_len / 2)
                x2_out = int(pos[0] + border_len / 2)
                y1_out = int(pos[1] - border_len / 2)
                y2_out = int(pos[1] + border_len / 2)
                # Draw grids
                cv2.rectangle(
                    base_img,
                    (x1_out, y1_out),
                    (x2_out, y2_out),
                    (0, 0, 0),
                    thickness=2
                )
                is_max = [0 for _ in q_act]
                argmax_idx = np.argmax(q_act)
                is_max[argmax_idx] = 1
                # 0-up
                poses = np.array(
                    [(x1_out, y1_out), pos, (x2_out, y1_out)],
                    np.int32
                )
                base_img = self.__draw_a_piece(base_img, [q_act[0], q_act_[0]], max_q, poses, is_max=is_max[0])
                # 1-down
                poses = np.array(
                    [(x1_out, y2_out), pos, (x2_out, y2_out)],
                    np.int32
                )
                base_img = self.__draw_a_piece(base_img, [q_act[1], q_act_[1]], max_q, poses, is_max=is_max[1])
                # 2-right
                poses = np.array(
                    [(x2_out, y1_out), pos, (x2_out, y2_out)],
                    np.int32
                )
                base_img = self.__draw_a_piece(base_img, [q_act[2], q_act_[2]], max_q, poses, is_max=is_max[2])
                # 3-left
                poses = np.array(
                    [(x1_out, y1_out), pos, (x1_out, y2_out)],
                    np.int32
                )
                base_img = self.__draw_a_piece(base_img, [q_act[3], q_act_[3]], max_q, poses, is_max=is_max[3])

                # Draw state
                if i == 0:
                    cv2.putText(
                        base_img, str(j), (int(h_p / 2), row),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )

        return base_img

    def __draw_a_piece(self, base_img, q, max_q, poses, is_max=0):
        # 画三角形
        cv2.polylines(
            base_img,
            [poses],
            True,
            (0, 0, 0),
            thickness=1
        )

        # 填充颜色
        pixel = int((1 - q[1] / (max_q+1e-8)) * 255)
        cv2.fillPoly(base_img, [poses], (pixel, pixel, pixel))

        poses_array = np.array(poses)
        max_x, min_x = max(poses_array[:, 0]), min(poses_array[:, 0])
        max_y, min_y = max(poses_array[:, 1]), min(poses_array[:, 1])

        # 写入Q值
        cv2.putText(
            base_img, str(round(q[0], 2)),
            (int((max_x+min_x)/2-15), int((max_y+min_y)/2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 255, 0) if is_max == 0 else (0, 0, 255),
            1, cv2.LINE_AA
        )
        return base_img

    def __draw_v(self, base_img, value, w_p, h_p, size, border_len):
        value_pi = value - np.min(value)
        max_v_pi = np.max(value_pi)

        for i in range(size):
            for j in range(size):
                pos = (int(border_len / 2 + i * border_len + w_p),
                       int(border_len / 2 + j * border_len + h_p))
                v_pi = value_pi[i + j * size]
                pixel = int((1 - v_pi / max_v_pi) * 255)
                # Draw grids
                cv2.rectangle(
                    base_img,
                    (int(pos[0] - border_len / 2), int(pos[1] - border_len / 2)),
                    (int(pos[0] + border_len / 2), int(pos[1] + border_len / 2)),
                    (pixel, pixel, pixel),
                    thickness=-1
                )
                cv2.putText(
                    base_img, str(round(v_pi, 1)), (pos[0] - 10, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), 1, cv2.LINE_AA
                )

        return base_img

    def close(self):
        if self.video is not None:
            self.video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video = None
