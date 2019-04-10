import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import random
from config import Config
from dataset import SyntheticData, label2point
from magic_point import SuperPointNet
from tensorboardX import SummaryWriter
import logging

from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


log = Logger('train.log', level='debug')


def output2points(output, alpha=0.001):
    output = np.exp(output.detach().numpy())  # Softmax.
    output = output / (np.sum(output, axis=0) + .00001)  # Should sum to 1.
    output = output[:-1, :, :]
    output = output.transpose(1, 2, 0)
    output = output.reshape((Hc, Wc, 8, 8))
    output = output.transpose(0, 2, 1, 3)
    output = output.reshape(H, W)
    # print(output.shape)
    # print(output[output > alpha])
    xs, ys = np.where(output > alpha)

    points = np.vstack((xs, ys)).T
    # if len(points) == 0:
    # print(f'there are {len(points)} points')
    # print(f"max output value is {np.max(output)}")
    # print(f"min output value is {np.min(output)}")
    return points


H = Config.H
W = Config.W
Hc = Config.Hc
Wc = Config.Wc

writer = SummaryWriter()

dataset_root = Config.dataset_root

train_csv = '/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/model/training.csv'
test_csv = '/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/model/test.csv'
val_csv = '/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/model/validation.csv'

model_save_path = '/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/model/result'

batch_size = 64

log.logger.info("loading data...")

train_data = SyntheticData(train_csv, dataset_root)
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)
test_data = SyntheticData(test_csv, dataset_root)
test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)
log.logger.info("Done")
log.logger.info("Loading model...")
net = SuperPointNet()
log.logger.info("done")
log.logger.info(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.001)

n_epoch = 200000
running_loss = 0.0

last_test_loss = 999999

net.train()
iter_num = 0
log.logger.info("Start training")
for epoch in range(n_epoch):
    for i, sample in enumerate(train_loader):
        imgs = sample['img'].view((-1, 1, H, W))
        labels = sample['label']

        outputs = net(imgs)
        loss = criterion(outputs, labels)
        #         for i in range(Hc):
        #             for j in range(Wc):
        #                 # print(outputs[:,:,i,j].shape,labels[:,i,j].shape)
        #                 loss += criterion(outputs[:,:,i,j],labels[:,i,j])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 1000 == 0:
            iter_num += 1
            log.logger.info(f"epoch:{epoch + 1},batch:{i + 1},AVG.loss:{running_loss / 1000}")
            writer.add_scalar('data/running_loss', running_loss, iter_num)
            running_loss = 0.0

            for j in range(8):
                img = imgs[j].detach().numpy()
                output = outputs[j]
                img = np.squeeze(img)
                points = output2points(output)
                plt.figure()
                plt.imshow(img)
                plt.axis("off")
                plt.scatter(points[:, 1], points[:, 0])
                plt.savefig(f'sample_output/sample_output_epoch{epoch + 1}_iter{i + 1}.png')
                plt.close('all')
                # print('save sample to ./sample_output/sample_output*.png')

            save_path = os.path.join(model_save_path, f"epoch{epoch + 1}_iter{i + 1}")
            torch.save(net, save_path)
            log.logger.info(f"save model to {save_path}")

            # calculate test loss
            test_loss = 0.0
            with torch.no_grad():
                for i, sample in enumerate(test_loader):
                    imgs = sample['img'].view((-1, 1, H, W))
                    labels = sample['label']
                    outputs = net(imgs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * imgs.shape[0]
                log.logger.info(f"AVG. test loss:{test_loss / len(test_data)}")
            writer.add_scalar('data/test_loss', test_loss, iter_num)
    if iter_num == 200:
        break

writer.export_scalars_to_json("data/all_scalars.json")
writer.close()
