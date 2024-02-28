import copy

import cv2
import numpy as np


class CVRender:
    def __init__(self, env, width=1200, height=1200, padding=0.05):
        self.video = cv2.VideoWriter(
            'figs/snake.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            8,
            (width, height)
        )

        self.width = width
        self.height = height
        self.env = env

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
            # Draw state
            cv2.putText(
                base_image, str(i), (column, int(h_p/2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

            for j in range(size):
                row = int(border_len / 2 + j * border_len + h_p)
                pos = (column, row)
                idx = i + j * size
                self.pos_dict[idx] = pos

                # Draw grids
                cv2.rectangle(
                    base_image,
                    (int(pos[0] - border_len / 2), int(pos[1] - border_len / 2)),
                    (int(pos[0] + border_len / 2), int(pos[1] + border_len / 2)),
                    (0, 0, 0),
                    thickness=2
                )

                # Draw targets
                if idx in env.targets:
                    cv2.circle(base_image, pos, 10, (0, 255, 0), thickness=-1)
                if idx == env.start:
                    cv2.circle(base_image, pos, 10, (255, 0, 0), thickness=2)

                # Draw state
                if i == 0:
                    cv2.putText(
                        base_image, str(j), (int(h_p/2), row),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
        # Draw ladders
        for key1, [values1, _] in env.ladders.items():
            pos = self.pos_dict[key1]
            cv2.circle(base_image, pos, 10, (0, 0, 0), thickness=2)

            for v in values1:
                cv2.line(base_image, pos, self.pos_dict[v], (0, 0, 0), thickness=1)

        self.base_img = base_image
        cv2.imwrite('figs/base_image.png', base_image)

    def draw(self, show=False, mode='algo', last_pos=None):
        width, height = self.width, self.height
        base_img = copy.deepcopy(self.base_img)

        # Text: mode (MC/TD/PI/VI)
        cv2.putText(
            base_img,
            mode,
            (40, height-40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

        pos = self.pos_dict[self.env.pos]
        cv2.circle(base_img, pos, 10, (255, 0, 0))

        cv2.putText(
            base_img,
            self.env.env_info,
            (width + 20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

        if last_pos is not None:
            for i, (pos, text) in enumerate(last_pos[::-1]):
                pos = self.pos_dict[pos]
                radius = 20 - 2 * (i + 1)
                thickness = 2 if i == 0 else 1
                if radius >= 1:
                    cv2.circle(base_img, pos, radius, (255, 0, 0), thickness=thickness)

                cv2.putText(
                    base_img, text,
                    (width + 50, (i + 2) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA
                )

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
