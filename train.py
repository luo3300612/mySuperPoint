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
from dataloader import SyntheticData, label2point, get_loader
from models.magic_point import MagicPoint
from tensorboardX import SummaryWriter
from utils import OutPutUtil
import argparse
import sys
from opts import parse_opts

writer = SummaryWriter()


def train(model, device, train_loader, optimizer, epoch, logger,opt):
    model.train()
    running_loss = 0.0
    H = opt['H']
    W = opt['W']
    for i, sample in enumerate(train_loader):
        imgs = sample['img'].view((-1, 1, H, W)).to(device)
        labels = sample['label'].to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    logger.speak(f"epoch:{epoch + 1},AVG.loss:{running_loss / 1000}")
    writer.add_scalar('data/running_loss', running_loss / 1000, epoch + 1)

    # save model
    save_path = os.path.join(opt['save-path'], f"epoch{epoch + 1}")
    torch.save(model, save_path)
    logger.speak(f"save model to {save_path}")


def test(model, device, test_loader, logger,opt):
    model.eval()
    test_loss = 0.0
    H = opt['H']
    W = opt['W']
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            imgs = sample['img'].view((-1, 1, H, W)).to(device)
            labels = sample['label'].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * imgs.shape[0]
        logger.speak(f"AVG. test loss:{test_loss / len(test_loader.dataset)}")
    writer.add_scalar('data/test_loss', test_loss / len(test_loader.dataset), epoch + 1)


if __name__ == '__main__':

    opt = parse_opts()
    opt = vars(opt)

    logger = OutPutUtil(log_file=os.path.join(opt['save-path'], opt['id']))

    use_cuda = not opt['no-cuda'] and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(opt['seed'])

    # load path

    batch_size = opt.batch_size

    # load data
    logger.speak("loading data...")
    train_loader = get_loader(opt, 'train', logger)
    test_loader = get_loader(opt, 'test', logger)
    logger.speak("Done")

    # load model
    logger.speak("Loading model...")
    net = MagicPoint()
    logger.speak("done")
    logger.speak(net)

    # prepare training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=opt.lr)
    n_epoch = opt['n_epoch']

    logger.speak("Start training")
    for epoch in range(n_epoch):
        # train
        train(net, device, train_loader, optimizer, epoch, logger, opt)
        # test
        test(net, device, test_loader, logger, opt)
