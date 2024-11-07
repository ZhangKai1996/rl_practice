import time
import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.ma.core import inner


inner_radius = 8
radius = 15
subtitle_size = 0.75
subtitle_color = (0, 0, 0)
font_thickness = 1
outer_font_size = 1.0
outer_font_color = (0, 0, 0)
inner_font_size = 0.75
inner_font_color = (150, 150, 150)


class EnvRender:
    def __init__(self, env, width=1200, height=1200, padding=0.05):
        self.video = cv2.VideoWriter(
            'figs/snake_{}.avi'.format(int(time.time())),
            cv2.VideoWriter_fourcc(*'MJPG'),
            8, (width, height)
        )
        self.width = width
        self.height = height
        self.padding = padding
        self.env = env
        self.base_img = None
        self.__initialize(env, height, width, padding)

    def __initialize(self, env, height, width, padding):
        base_image = np.ones((height, width, 3), np.uint8) * 255
        w_p = int(width * padding)
        h_p = int(height * padding)
        size = env.size
        border_len = int((width - w_p * 2) / size)

        self.pos_dict = {}
        for i in range(size):
            column = int(border_len / 2 + i * border_len + w_p)
            # Draw row number
            cv2.putText(base_image, str(i), (column, int(height - h_p / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        outer_font_size, outer_font_color, font_thickness,
                        cv2.LINE_AA)
            for j in range(size):
                row = int(border_len / 2 + j * border_len + h_p)
                pos = (column, row)
                idx = i + j * size
                self.pos_dict[idx] = pos
                # Draw grids
                thickness = -1 if idx in env.barriers else 2
                cv2.rectangle(base_image,
                             (int(pos[0] - border_len / 2), int(pos[1] - border_len / 2)),
                             (int(pos[0] + border_len / 2), int(pos[1] + border_len / 2)),
                              (0, 0, 0), thickness=thickness)
                # Draw targets
                if idx in env.coins:
                    cv2.circle(base_image, pos, radius, (0, 255, 0), thickness=2)
                elif idx == env.start:
                    cv2.circle(base_image, pos, radius, (255, 0, 0), thickness=2)
                elif idx in env.mud:
                    cv2.circle(base_image, pos, radius, (255, 0, 255), thickness=2)
                elif idx in env.land:
                    cv2.circle(base_image, pos, radius, (255, 255, 0), thickness=2)
                cv2.putText(base_image, str(idx),
                            (column+int(border_len/4), row+int(border_len/4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            outer_font_size, outer_font_color, font_thickness,
                            cv2.LINE_AA)
                # Draw column number
                if i == 0:
                    cv2.putText(base_image, str(j), (int(w_p / 2), row),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                outer_font_size, outer_font_color, font_thickness,
                                cv2.LINE_AA)
        # Draw ladders
        for key1, [values1, _] in env.ladders.items():
            pos = self.pos_dict[key1]
            cv2.circle(base_image, pos, inner_radius, (0, 0, 0), thickness=2)
            for v in values1:
                cv2.line(base_image, pos, self.pos_dict[v], (0, 0, 0), thickness=1)
        # Draw planned path
        points = [env.start, ] + env.land + env.coins
        for i, key1 in enumerate(points[:-1]):
            key2 = points[i+1]
            cv2.line(base_image, self.pos_dict[key1], self.pos_dict[key2],
                     inner_font_color, thickness=1)

        self.base_img = base_image
        self.img = copy.deepcopy(base_image)
        cv2.imwrite('figs/base_image.png', base_image)

    def draw(self, refresh=False, show=False, mode=None):
        env = self.env
        width, height = self.width, self.height
        if refresh:
            self.base_img = copy.deepcopy(self.img)
        base_img = copy.deepcopy(self.base_img)
        # Text: mode (MC/TD/PI/VI)
        if mode is not None:
            delta_w = int(width * self.padding)
            delta_h = int(height * self.padding / 2)
            cv2.putText(base_img, mode, (delta_w, delta_h),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        subtitle_size, subtitle_color, font_thickness,
                        cv2.LINE_AA)
        pos = self.pos_dict[env.pos]
        cv2.circle(base_img, pos, inner_radius, (0, 0, 255), thickness=-1)
        last_pos = self.pos_dict[env.last_pos]
        cv2.line(self.base_img, pos, last_pos, (0, 0, 255), thickness=1)
        self.video.write(base_img)
        if show:
            cv2.imshow('basic image', base_img)
            if cv2.waitKey(0) == 113:
                cv2.destroyAllWindows()

    def close(self):
        if self.video is not None:
            self.video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video = None


class ValueRender:
    def __init__(self, env, algo='algo', width=1200, height=1200, padding=0.05):
        self.width = width
        self.height = height
        self.padding = padding
        # self.video = cv2.VideoWriter(
        #     'figs/value_{}.avi'.format(algo),
        #     cv2.VideoWriter_fourcc(*'MJPG'),
        #     30,
        #     (width * 2, height)
        # )
        self.env = env
        self.base_img = np.ones((height, width, 3), np.uint8) * 255

    def draw(self, values: dict, algo='algo'):
        w_p = int(self.width * self.padding)
        h_p = int(self.height * self.padding)
        size = self.env.size
        border_len = int((self.width - w_p * 2) / size)

        images = []
        if 'v' in values.keys():
            base_img = self.__draw_v(values['v'], w_p, h_p, size, border_len)
            images.append(base_img)
        if 'r' in values.keys():
            base_img = self.__draw_v(values['r'], w_p, h_p, size, border_len)
            images.append(base_img)
        if 'q' in values.keys():
            base_img = self.__draw_q(values['q'], w_p, h_p, size, border_len)
            images.append(base_img)
        if 'pi' in values.keys():
            base_img = self.__draw_q(values['pi'], w_p, h_p, size, border_len)
            images.append(base_img)
        if len(images) > 0:
            if len(images) == 4:
                frame = np.vstack([
                    np.hstack(images[:2]),
                    np.hstack(images[2:])
                ])
            else:
                frame = np.vstack(images)
            cv2.imwrite('figs/value_{}_{}.png'.format(algo, int(time.time())), frame)

    def __draw_a_piece(self, base_img, poses, q_pi=None, is_max=0):
        # 画三角形
        cv2.polylines(base_img, [poses], True, (0, 0, 0))
        [p1, p0, p2] = poses
        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)
        if q_pi is not None:
            cv2.putText(base_img, str(round(q_pi, 2)),
                        (int((p0[0] + x) / 2), int((p0[1] + y) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        inner_font_size, inner_font_color, font_thickness,
                        cv2.LINE_AA)
        # 画红线
        if is_max != 0:
            cv2.line(base_img, p0, (x, y), (0, 0, 255))
        return base_img

    def __draw_q(self, value, w_p, h_p, size, border_len):
        base_img = copy.deepcopy(self.base_img)
        for i in range(size):
            column = int(border_len / 2 + i * border_len + w_p)
            # Draw state
            cv2.putText(base_img, str(i), (column, int(h_p / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        outer_font_size, outer_font_color, font_thickness,
                        cv2.LINE_AA)
            for j in range(size):
                row = int(border_len / 2 + j * border_len + h_p)
                pos = (column, row)
                idx = i + j * size

                x1_out = int(pos[0] - border_len / 2)
                x2_out = int(pos[0] + border_len / 2)
                y1_out = int(pos[1] - border_len / 2)
                y2_out = int(pos[1] + border_len / 2)

                # Draw targets
                if idx in self.env.coins:
                    cv2.circle(base_img, pos, 10, (0, 255, 0), thickness=2)
                elif idx == self.env.start:
                    cv2.circle(base_img, pos, 10, (255, 0, 0), thickness=2)
                elif idx in self.env.mud:
                    cv2.circle(base_img, pos, 10, (255, 0, 255), thickness=2)
                elif idx in self.env.land:
                    cv2.circle(base_img, pos, 10, (255, 255, 0), thickness=2)

                q_act = value[idx]
                is_max = [int(q_ == max(q_act)) for q_ in q_act]
                # 1-up
                poses = np.array([(x1_out, y1_out), pos, (x2_out, y1_out)], np.int32)
                base_img = self.__draw_a_piece(base_img, poses, q_pi=q_act[1], is_max=is_max[1])
                # 2-down
                poses = np.array([(x1_out, y2_out), pos, (x2_out, y2_out)], np.int32)
                base_img = self.__draw_a_piece(base_img, poses, q_pi=q_act[2], is_max=is_max[2])
                # 0-right
                poses = np.array([(x2_out, y1_out), pos, (x2_out, y2_out)], np.int32)
                base_img = self.__draw_a_piece(base_img, poses, q_pi=q_act[0], is_max=is_max[0])
                # 3-left
                poses = np.array([(x1_out, y1_out), pos, (x1_out, y2_out)], np.int32)
                base_img = self.__draw_a_piece(base_img, poses, q_pi=q_act[3], is_max=is_max[3])

                # Draw state
                if i == 0:
                    cv2.putText(base_img, str(j), (int(h_p / 2), row),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                outer_font_size, outer_font_color, font_thickness,
                                cv2.LINE_AA)
        return base_img

    def __draw_v(self, value_pi, w_p, h_p, size, border_len, reg=False):
        base_img = copy.deepcopy(self.base_img)
        env = self.env

        if reg:
            value_pi = value_pi - np.min(value_pi)
            max_v_pi = np.max(value_pi)
            if max_v_pi > 0:
                value_pi /= max_v_pi

        for i in range(size):
            column = int(border_len / 2 + i * border_len + w_p)
            # Draw state
            cv2.putText(base_img, str(i), (column, int(h_p / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        outer_font_size, outer_font_color, font_thickness,
                        cv2.LINE_AA)
            for j in range(size):
                row = int(border_len / 2 + j * border_len + h_p)
                pos = (column, row)
                idx = i + j * size
                v_pi = value_pi[i + j * size]
                # Draw grids
                cv2.rectangle(
                    base_img,
                    (int(pos[0] - border_len / 2), int(pos[1] - border_len / 2)),
                    (int(pos[0] + border_len / 2), int(pos[1] + border_len / 2)),
                    (0, 0, 0),
                    thickness=-1 if idx in env.barriers else 1
                )
                cv2.putText(base_img, '{:>5.2f}'.format(round(v_pi, 2)),
                            (pos[0] - 10, pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            inner_font_size, inner_font_color, font_thickness,
                            cv2.LINE_AA)
                # Draw targets
                if idx in env.coins:
                    cv2.circle(base_img, pos, 10, (0, 255, 0), thickness=2)
                elif idx == env.start:
                    cv2.circle(base_img, pos, 10, (255, 0, 0), thickness=2)
                elif idx in env.mud:
                    cv2.circle(base_img, pos, 10, (255, 0, 255), thickness=2)
                elif idx in env.land:
                    cv2.circle(base_img, pos, 10, (255, 255, 0), thickness=2)
                # Draw state
                if i == 0:
                    cv2.putText(base_img, str(j), (int(h_p / 2), row),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                outer_font_size, outer_font_color, font_thickness,
                                cv2.LINE_AA)
        return base_img

    def close(self):
        self.env = None
        self.base_img = None
        # if self.video is not None:
        #     self.video.release()
        #     cv2.waitKey(1) & 0xFF
        #     cv2.destroyAllWindows()
        #     self.video = None


def regularize(value):
    value_pi = value - np.min(value)
    max_v_pi = np.max(value_pi)
    if max_v_pi > 0:
        value_pi /= max_v_pi
    return value_pi


def save_animation(values, r, algo, reg=True):
    fig = plt.figure()  # 创建一个3D图形对象
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if reg:
        r = regularize(r)

    def init():  # 初始化函数，创建空白图形
        ax1.cla()  # 清除当前图形
        ax1.set_title('Reward')
        num = int(np.sqrt(len(r)))
        x = np.linspace(0, num - 1, num)
        y = np.linspace(0, num - 1, num)
        x_grid, y_grid = np.meshgrid(x, y)
        z = np.array(r).reshape(num, num)
        ax1.plot_surface(x_grid, y_grid, z, cmap='rainbow')
        return ax1, ax2

    def update(frame):  # 动画更新函数，用于旋转立方体
        ax2.cla()  # 清除当前图形
        ax2.set_title(f'Value (Frame {frame})')

        value = values[frame]
        if reg:
            value = regularize(value)
        num = int(np.sqrt(len(value)))
        x = np.linspace(0, num - 1, num)
        y = np.linspace(0, num - 1, num)
        x_grid, y_grid = np.meshgrid(x, y)
        z = value.reshape(num, num)
        # print(frame, x_grid.shape, y_grid.shape, z.shape)
        ax2.plot_surface(x_grid, y_grid, z, cmap='rainbow')
        return ax1, ax2

    frames = len(values)  # 动画帧数
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
    # 在创建FuncAnimation对象之后，添加以下代码保存动画为GIF文件
    ani.save(
        'figs/value_{}_{}.gif'.format(algo, int(time.time())),
        writer='pillow',
        fps=30
    )
    # 显示动画
    # plt.show()
