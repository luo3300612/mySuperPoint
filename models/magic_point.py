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


class MagicPoint(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(MagicPoint, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1a = torch.nn.BatchNorm2d(c1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1b = torch.nn.BatchNorm2d(c1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn2a = torch.nn.BatchNorm2d(c2)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.bn2b = torch.nn.BatchNorm2d(c2)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn3a = torch.nn.BatchNorm2d(c3)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bn3b = torch.nn.BatchNorm2d(c3)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.bn4a = torch.nn.BatchNorm2d(c4)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.bn4b = torch.nn.BatchNorm2d(c4)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = torch.nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        self.bnPb = torch.nn.BatchNorm2d(65)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.bn1a(x)
        x = self.relu(self.conv1b(x))
        x = self.bn2a(x)
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.bn2a(x)
        x = self.relu(self.conv2b(x))
        x = self.bn2b(x)
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.bn3a(x)
        x = self.relu(self.conv3b(x))
        x = self.bn3b(x)
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.bn4a(x)
        x = self.relu(self.conv4b(x))
        x = self.bn4b(x)

        # Detector Head.
        cPa = self.bnPa(self.relu(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        #         cDa = self.relu(self.convDa(x))
        #         desc = self.convDb(cDa)
        #         dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        #         desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi  # , desc