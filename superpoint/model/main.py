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


def output2points(output, alpha=0.5):
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
    if len(points) == 0:
        print(f'there are {len(points)} points')
        print(f"max output value is {np.max(output)}")
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

batch_size = 16

train_data = SyntheticData(val_csv, dataset_root)
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)
test_data = SyntheticData(test_csv, dataset_root)
test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)

net = SuperPointNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=0.01)

n_epoch = 10
running_loss = 0.0

last_test_loss = 999999

net.train()
count = 0
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
        running_loss += loss.data

        if i % 10 == 9:
            count += 1
            print(f"epoch:{epoch + 1},batch:{i + 1},AVG.loss:{running_loss / batch_size / 10}")
            writer.add_scalar('data/running_loss', running_loss, count)
            running_loss = 0.0

    # sample image
    for i in range(8):
        img = imgs[i].detach().numpy()
        output = outputs[i]
        img = np.squeeze(img)
        points = output2points(output)
        plt.imshow(img)
        plt.axis("off")
        plt.scatter(points[:, 1], points[:, 0])
        plt.savefig(f'sample_output/sample_output{i}.png')
        print('save sample to ./sample_output/sample_output*.png')

    save_path = os.path.join(model_save_path, f"epoch{epoch}")
    torch.save(net, save_path)
    print(f"save model to {save_path}")

    # calculate test loss
    # test_loss = 0.0
    # with torch.no_grad():
    #     for i, sample in enumerate(test_loader):
    #         imgs = sample['img'].view((-1, 1, H, W))
    #         labels = sample['label']
    #         outputs = net(imgs)
    #         loss = criterion(outputs, labels)
    #         test_loss += loss.data
    # if test_loss < last_test_loss:
    #     last_test_loss = test_loss
    #     print(f"AVG. test loss:{test_loss / len(test_data)} ↑")
    #     save_path = os.join(model_save_path, f"epoch{epoch}")
    #     torch.save(net, save_path)
    #     print(f"save model to {save_path}")
    # else:
    #     print(f"AVG. test loss:{test_loss / len(test_data)}")
    # writer.add_scalar('data/test_loss', test_loss, epoch+1)

writer.export_scalars_to_json("data/all_scalars.json")
writer.close()
