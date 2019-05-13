import os
import tqdm
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from PIL import Image



sys.path.append('/home/luo3300612/Workspace/PycharmWS/mySuperPoint')

from superpoint.model.config import Config

dataset_root = Config.dataset_root

first_dir_names = ['draw_checkerboard',
                   'draw_cube',
                   'draw_ellipses',
                   'draw_lines',
                   'draw_multiple_polygons',
                   'draw_polygon',
                   'draw_star',
                   'draw_stripes',
                   'gaussian_noise']

second_images_dir_name = 'images'
second_pts_dir_name = 'points'

dataset_dir_name = {'train': 'training', 'test': 'test', 'val': 'validation'}

H = Config.H
W = Config.W
Hc = Config.Hc
Wc = Config.Wc


def gen_csv():
    for value in dataset_dir_name.values():
        print(f'生成{value}.csv')
        df = pd.DataFrame(columns=['imgs_path', 'pts_path'])
        for first_dir_name in first_dir_names:
            imgs_prefix = os.path.join(first_dir_name,
                                       second_images_dir_name,
                                       value)
            for _, _, imgs in os.walk(os.path.join(dataset_root,
                                                   first_dir_name,
                                                   second_images_dir_name,
                                                   value)):
                imgs_ret = sorted(imgs)
                imgs_ret = [os.path.join(imgs_prefix, img) for img in imgs_ret]
                imgs_ret = pd.Series(imgs_ret)

            pts_prefix = os.path.join(first_dir_name,
                                      second_pts_dir_name,
                                      value)
            for _, _, pts in os.walk(os.path.join(dataset_root,
                                                  first_dir_name,
                                                  second_pts_dir_name,
                                                  value)):
                pts_ret = sorted(pts)
                pts_ret = [os.path.join(pts_prefix, pt) for pt in pts_ret]
                pts_ret = pd.Series(pts_ret)

            ret = pd.DataFrame({'imgs_path': imgs_ret, 'pts_path': pts_ret})
            df = df.append(ret)
        print('检查路径对齐是否正确...')

        for i in tqdm(range(len(df))):
            img_path = df.iloc[i, :]['imgs_path']
            pt_path = df.iloc[i, :]['pts_path']
            if img_path.split('/')[-1].split('.')[0] != pt_path.split('/')[-1].split('.')[0]:
                print('error,i=', i)
                break
        else:
            print('OK')
            df.to_csv(f'{value}.csv')


class SyntheticData(Dataset):
    def __init__(self, csv_file, dataset_root, save_point=False, only_point=False):
        self.csv = pd.read_csv(csv_file)
        self.dataset_root = dataset_root
        self.save_point = save_point
        self.only_point = only_point

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        item = self.csv.iloc[idx]
        img_path = os.path.join(self.dataset_root, item['imgs_path'])
        pt_path = os.path.join(self.dataset_root, item['pts_path'])
        img = plt.imread(img_path)
        pt = np.load(pt_path)
        if self.only_point:
            sample = {'img': img, 'pt': pt}
        elif self.save_point:
            sample = {'img': img, 'label': point2label(pt), 'pt': pt}
        else:
            sample = {'img': img, 'label': point2label(pt)}
        return sample


class COCO(Dataset):
    def __init__(self, dataset_root, transform=None):
        self.dataset_root = dataset_root
        self.imgs = os.listdir(dataset_root)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dataset_root, self.imgs[idx]))
        if self.transform:
            return self.transform(img)
        else:
            return img


def point2label(pts, binary=False):
    label = np.zeros((H, W), dtype=int)
    pts = pts.astype(int)
    #     print(pts)
    #     print(pts.shape)
    label[pts[:, 0], pts[:, 1]] = 1
    if binary:
        return label
    label = label.reshape((Hc, 8, Wc, 8))
    label = label.transpose((0, 2, 1, 3))
    label = label.reshape((Hc, Wc, 64))
    label = np.concatenate((2 * label, np.ones((Hc, Wc, 1), dtype=int)), axis=2)
    label = np.argmax(label, axis=2)
    return label


def label2point(label):
    ret = []
    for i in range(Hc):
        for j in range(Wc):
            if label[i, j] != 64:
                x = label[i, j] // 8 + i * 8
                y = label[i, j] % 8 + j * 8
                ret.append([x, y])
    return np.array(ret)


def visulize(img, label=None, pt=None, pts_color='b'):
    img = img.squeeze()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if label is not None:
        for i in range(Hc):
            for j in range(Wc):
                x, y = i * 8, j * 8
                if label[i, j] == 64:
                    continue
                k, l = label[i, j] // 8, int(label[i, j]) % 8
                plt.gca().add_patch(plt.Rectangle((y, x), 8, 8, color='r', fill=False, linewidth=2))
    if pt is not None and len(pt) != 0:
        try:
            assert pt.shape[1] == 2
        except AssertionError as err:
            print("Dim of pts not correct")
            raise
        plt.scatter(pt[:, 1], pt[:, 0], color=pts_color)
    plt.show()

# if __name__ == '__main__':
#     gen_csv()
