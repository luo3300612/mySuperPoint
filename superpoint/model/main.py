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
from log import Logger
from utils import output2points
import argparse

log = Logger('train.log', level='debug')
writer = SummaryWriter()

H = Config.H
W = Config.W
Hc = Config.Hc
Wc = Config.Wc


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, sample in enumerate(train_loader):
        imgs = sample['img'].view((-1, 1, H, W)).to(device)
        labels = sample['label'].to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    log.logger.info(f"epoch:{epoch + 1},AVG.loss:{running_loss / 1000}")
    writer.add_scalar('data/running_loss', running_loss, epoch + 1)

    # save model
    save_path = os.path.join(model_save_path, f"epoch{epoch + 1}")
    torch.save(model, save_path)
    log.logger.info(f"save model to {save_path}")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            imgs = sample['img'].view((-1, 1, H, W)).to(device)
            labels = sample['label'].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * imgs.shape[0]
        log.logger.info(f"AVG. test loss:{test_loss / len(test_data)}")
    writer.add_scalar('data/test_loss', test_loss, epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Superpoint Pytorch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    # load path
    dataset_root = Config.dataset_root
    train_csv = Config.train_csv
    test_csv = Config.test_csv
    val_csv = Config.val_csv
    model_save_path = Config.model_save_path

    batch_size = args.batch_size

    # load data
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

    # load model
    log.logger.info("Loading model...")
    net = SuperPointNet()
    log.logger.info("done")
    log.logger.info(net)

    # prepare training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=args.lr)
    n_epoch = args.epochs

    log.logger.info("Start training")
    for epoch in range(n_epoch):
        # train
        train(net, device, train_loader, optimizer, epoch)
        # test
        test(net, device, test_loader)

    writer.export_scalars_to_json("data/all_scalars.json")
    writer.close()
