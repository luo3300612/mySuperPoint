import cv2
import numpy as np
import os
import random

path_to_dataset = "/home/luo3300612/Workspace/PycharmWS/SuperPoint/data/synthetic_shapes_v6"

dir_image = "images"
dir_point = "points"
dir_train = "training"
dir_test = "test"
dir_validation = "validation"

ret = os.walk(path_to_dataset)
_, dirs, _ = next(ret)

m = 120  # height of img
n = 160  # width of img


def draw_point(img, points):
    """
    draw point only for points read from .npy of dataset
    """
    for point in points:
        cv2.circle(img, (point[1], point[0]), 1, (0, 0, 255), 4)


def img_join(imgs, size):
    """
    join imgs
    for example, join 9 imgs to get a 3*3 grid
    """
    assert len(imgs) == size[0] * size[1]
    combin = np.zeros((m * size[0], n * size[1], 3), dtype=np.uint8)
    imgs_iter = iter(imgs)
    for i in range(size[0]):
        for j in range(size[1]):
            combin[i * m:i * m + m, j * n:j * n + n, :] = next(imgs_iter)
    return combin


def check_shape(shape):
    """
    gen a 3 * 3 joint img of shape with keypoints
    """
    for dir in dirs:
        if shape in dir:
            dir_name = dir
            break
    else:
        raise ValueError("There is no " + shape)

    path2dir = os.path.join(path_to_dataset, dir_name)
    path2img = os.path.join(path2dir, dir_image, dir_train)
    path2point = os.path.join(path2dir, dir_point, dir_train)

    _, _, files = next(os.walk(path2img))

    imgs = []
    for i in range(9):
        file = random.choice(files)
        file_path = os.path.join(path2img, file)
        point_path = os.path.join(path2point, file.split('.')[0] + '.npy')
        img = cv2.imread(file_path)

        imgs.append(img)

        points = np.load(point_path)
        points = points.astype(np.uint8)
        print(f"There are {len(points)} keypoints")
        draw_point(img, points)

    combine = img_join(imgs, (3, 3))

    return combine


def gen_sample():
    """
    choose 1 from each kind of shape and combine them to a 3*3 img
    """
    imgs = []
    for i, dir in enumerate(dirs):
        path2train = os.path.join(path_to_dataset, dir, dir_image, dir_train)
        path_dirs_files = os.walk(path2train)

        _, _, files = next(path_dirs_files)
        file = random.choice(files)
        file_path = os.path.join(path2train, file)

        img = cv2.imread(file_path)
        imgs.append(img)

        point_path = os.path.join(path_to_dataset, dir, dir_point, dir_train, file.split('.')[0] + ".npy")
        points = np.load(point_path)
        points = points.astype(np.uint8)

        draw_point(img, points)

    combine = img_join(imgs, (3, 3))

    return combine

    # save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "samples", "sample.png")
    # cv2.imwrite(save_path, combine)


# check_shape("line")  # checkerboard cube ellipse line multiple polygon polygon star stripes gaussian noise


if __name__ == '__main__':
    c = 0
    while c != 113:
        combine = gen_sample()
        cv2.imshow("Window1", combine)
        c = cv2.waitKey(0)
