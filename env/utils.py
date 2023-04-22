import copy

import cv2
import numpy as np


class CVRender:
    def __init__(self, env, width=800, height=800, padding=0.05):
        self.video = cv2.VideoWriter('snake.avi',
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     8,
                                     (width, height))
        self.width = width
        self.height = height
        self.env = env
        # 创建一个的白色画布，RGB(255,255,255)为白色
        base_image = np.ones((height+200, width, 3), np.uint8) * 255
        w_p = int(width * padding)
        h_p = int(height * padding)
        size = env.size
        border_len = int((width - w_p * 2) / size)
        self.pos_dict = {}
        for i in range(size):
            for j in range(size):
                pos = (int(border_len / 2 + i * border_len + w_p),
                       int(border_len / 2 + j * border_len + h_p))
                idx = i + j * size
                self.pos_dict[idx] = pos
                cv2.rectangle(base_image,
                              (int(pos[0] - border_len / 2), int(pos[1] - border_len / 2)),
                              (int(pos[0] + border_len / 2), int(pos[1] + border_len / 2)),
                              (0, 0, 0))
                if idx == env.target:
                    cv2.circle(base_image, pos, 10, (0, 0, 255), thickness=-1)
                if idx in env.traps:
                    cv2.circle(base_image, pos, 10, (255, 0, 255), thickness=-1)

                cv2.putText(base_image, str(idx), pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (0, 0, 0), 1, cv2.LINE_AA)

        for key1, key2 in env.ladders.items():
            cv2.line(base_image, self.pos_dict[key1], self.pos_dict[key2], (0, 255, 0))

        self.base_img = base_image

    def draw(self, show=False, last_pos=None):
        base_img = copy.deepcopy(self.base_img)
        pos = self.pos_dict[self.env.pos]
        cv2.circle(base_img, pos, 10, (255, 0, 0))

        cv2.putText(base_img, self.env.env_info, (30, self.height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        if last_pos is not None:
            for i, (pos, text) in enumerate(last_pos[::-1]):
                pos = self.pos_dict[pos]
                radius = 10 - 2*(i+1)
                if radius >= 1:
                    cv2.circle(base_img, pos, radius, (255, 0, 0))

                cv2.putText(base_img, text, (50, self.height+(i+2)*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)
        self.video.write(base_img)

        if show:
            cv2.imshow('base image', base_img)
            if cv2.waitKey(0) == 113:
                cv2.destroyAllWindows()

    def close(self):
        if self.video is not None:
            self.video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video = None
