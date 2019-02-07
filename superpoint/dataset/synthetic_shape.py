import logging
import synthetic_dataset
import numpy as np
import os
from tqdm import tqdm
import cv2
import json
from pathlib import Path

class SyntheticShape:
    def __init__(self):
        pass

    def _init_dir(self, config):
        pass

    def gen_data(self, primitive, config):
        basedir = os.path.join(config['basedir'], primitive)
        logging.info('Generating tarfile for primitive {}.'.format(primitive))
        synthetic_dataset.set_random_state(np.random.RandomState(
            config['generation']['random_seed']))
        for split, size in config['generation']['split_sizes'].items():
            im_dir, pts_dir = [Path(basedir, i, split) for i in ['images', 'points']]

            print("im_dir:", im_dir)
            print("pts_dir:", pts_dir)
            print("confirm?(y/n)")
            c = input()
            if c != "y":
                exit()

            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(size), desc=split, leave=False):
                image = synthetic_dataset.generate_background(
                    config['generation']['image_size'],
                    **config['generation']['params']['generate_background'])
                points = np.array(getattr(synthetic_dataset, primitive)(
                    image, **config['generation']['params'].get(primitive, {})))
                points = np.flip(points, 1)  # reverse convention with opencv

                b = config['preprocessing']['blur_size']
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (points * np.array(config['preprocessing']['resize'], np.float)
                          / np.array(config['generation']['image_size'], np.float))
                image = cv2.resize(image, tuple(config['preprocessing']['resize'][::-1]),
                                   interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(str(Path(im_dir, '{}.png'.format(i))), image)
                np.save(str(Path(pts_dir, '{}.npy'.format(i))), points)


if __name__ == '__main__':
    with open("magic-point_shapes.json", 'r') as f:
        config = json.load(f)
        dataset = SyntheticShape()
        dataset.gen_data("draw_checkerboard", config)
