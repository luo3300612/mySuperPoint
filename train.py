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
import time
import datetime
from opts import parse_opts

writer = SummaryWriter()


def test(model, device, test_loader, logger, iteration, opt):
    model.eval()
    test_loss = 0.0
    H = opt.H
    W = opt.W
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            imgs = sample['img'].view((-1, 1, H, W)).to(device)
            labels = sample['label'].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.shape[0]
        logger.speak("AVG. test loss:{:.4f}".format(test_loss / len(test_loader.dataset)))
    writer.add_scalar('test_loss', test_loss / len(test_loader.dataset), iteration)

    torch.save(model.state_dict(), os.path.join(opt.save_path, opt.id, 'model{}.pth'.format(iteration)))
    model.train()


if __name__ == '__main__':

    opt = parse_opts()

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    if not os.path.exists(os.path.join(opt.save_path, opt.id)):
        os.mkdir(os.path.join(opt.save_path, opt.id))
    logger = OutPutUtil(log_file=os.path.join(opt.save_path, opt.id,"train.log"))

    logger.speak(vars(opt))
    use_cuda = not opt.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(opt.seed)

    # load path

    batch_size = opt.batch_size

    # load data
    logger.speak("loading data...")
    train_loader = get_loader(opt, 'train', logger)
    test_loader = get_loader(opt, 'test', logger)
    logger.speak("Done")

    # load model
    logger.speak("Loading model...")
    net = MagicPoint().to(device)
    logger.speak("done")
    logger.speak(net)

    # prepare training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=opt.lr)
    n_iter = opt.n_iter
    n_epoch = n_iter * batch_size / len(train_loader.dataset)

    logger.speak("Start training")
    net.train()
    iteration = 0
    running_loss = 0.0
    interval = 0
    H = opt.H
    W = opt.W
    start_print = time.time()
    start_epoch = time.time()
    epoch = 0
    while True:
        epoch += 1
        for i, sample in enumerate(train_loader):
            iteration += 1
            imgs = sample['img'].view((-1, 1, H, W)).to(device)
            labels = sample['label'].to(device)

            outputs = net(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.shape[0]
            interval += labels.shape[0]

            if iteration % opt.print_every == 0:
                logger.speak(
                    "Iter:{}, Loss:{:.4f}, Time:{:.4f}".format(iteration, running_loss / interval, time.time() - start_print))
                writer.add_scalar('running_loss', running_loss, iteration)
                start_print = time.time()
                interval = 0
                running_loss = 0

            if iteration % opt.eval_every == 0:
                test(net, device, test_loader, logger,iteration, opt)
            if iteration > n_iter:
                logger.speak("Done")
                break
        delta = time.time() - start_epoch
        remain_time = datetime.timedelta(seconds=delta * (n_epoch - epoch))
        logger.speak("Epoch {} Done, Time:{}, Estimated Complete in {}".format(epoch, delta, remain_time))
        start_epoch = time.time()
