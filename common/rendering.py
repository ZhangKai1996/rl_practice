import time
import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


inner_radius = 5
radius = 10
subtitle_size = 0.6
subtitle_color = (0, 0, 0)
font_thickness = 1
outer_font_size = 0.6
outer_font_color = (0, 0, 0)
inner_font_size = 0.4
inner_font_color = (150, 150, 150)


class EnvRender:
    def __init__(self, env, width=1200, height=1200, padding=0.05):
        self.env = env
        self.width = width
        self.height = height
        self.padding = padding

        self.video = cv2.VideoWriter(
            filename='figs/snake_{}.avi'.format(int(time.time())),
            fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
            fps=8, frameSize=(width, height)
        )

        self.base_img = None
        self.pos_dict = None
        self.initialize()

    def initialize(self):
        base_image = np.ones((self.height, self.width, 3), np.uint8) * 255
        w_p = int(self.width * self.padding)
        h_p = int(self.height * self.padding)
        border_len = int((self.width - w_p * 2) / self.env.size)

        self.pos_dict = {}
        for i in range(self.env.size):
            column = int(border_len / 2 + i * border_len + w_p)
            # Draw row number
            cv2.putText(
                img=base_image,
                text=str(i),
                org=(column, int(self.height - h_p / 2)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=outer_font_size,
                color=outer_font_color,
                thickness=font_thickness,
                lineType=cv2.LINE_AA)

            for j in range(self.env.size):
                row = int(border_len / 2 + j * border_len + h_p)
                idx = i + j * self.env.size
                self.pos_dict[idx] = (column, row)

                # Draw grids
                cv2.rectangle(
                    img=base_image,
                    pt1=(int(column - border_len / 2), int(row - border_len / 2)),
                    pt2=(int(column + border_len / 2), int(row + border_len / 2)),
                    color=(0, 0, 0),
                    thickness=-1 if idx in self.env.barriers else 2
                )

                # Draw elements
                if idx in self.env.coins:
                    cv2.circle(base_image, (column, row), radius, (0, 255, 0), thickness=2)
                elif idx == self.env.start:
                    cv2.circle(base_image, (column, row), radius, (255, 0, 0), thickness=2)
                elif idx in self.env.mud:
                    cv2.circle(base_image, (column, row), radius, (173, 222, 255), thickness=-1)
                elif idx in self.env.land:
                    cv2.circle(base_image, (column, row), radius, (255, 255, 0), thickness=1)

                # Draw column number
                if i == 0:
                    cv2.putText(
                        img=base_image,
                        text=str(j),
                        org=(int(w_p / 2), row),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=outer_font_size,
                        color=outer_font_color,
                        thickness=font_thickness,
                        lineType=cv2.LINE_AA
                    )
        self.base_img = base_image
        cv2.imwrite('figs/base_image.png', base_image)

    def draw(self, show=False, mode=None):
        base_img = copy.deepcopy(self.base_img)
        # Text: mode (MC/TD/PI/VI)
        if mode is not None:
            cv2.putText(
                img=base_img,
                text=mode,
                org=(int(self.width * self.padding), int(self.height * self.padding / 2)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=subtitle_size,
                color=subtitle_color,
                thickness=font_thickness,
                lineType=cv2.LINE_AA
            )

        pos = self.pos_dict[self.env.pos]
        cv2.circle(base_img, pos, inner_radius, (0, 0, 255), thickness=-1)
        last_pos = self.pos_dict[self.env.last_pos]
        cv2.line(self.base_img, pos, last_pos, (0, 0, 255), thickness=1)

        if show:
            cv2.imshow('basic image', base_img)
            if cv2.waitKey(0) == 113:
                cv2.destroyAllWindows()
        self.video.write(base_img)

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
