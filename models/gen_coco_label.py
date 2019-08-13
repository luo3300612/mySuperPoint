import torch
import torchvision.transforms as transforms
from models.dataset import COCO
from models.homography import  homographic_adaptation,heatmap2points
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize((240, 320)),
                                # transforms.Normalize((0.5,), (0.5,))
                                ])

train_dataset = COCO('/run/media/luo3300612/我是D盘~ o(*￣▽￣*)ブ/下载/迅雷下载/coco/train2014', transform=transform)

net = torch.load('/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/result/epoch120',
                     map_location='cpu')

Nh = 100
top_k = 100
for idx in range(len(train_dataset)):
    img = train_dataset[5]
    heatmap = homographic_adaptation(net,np.array(img),Nh)
    pts = heatmap2points(heatmap,conf_thresh=0.015, nms_dist=4, border_remove=0)
    pts = pts[:,pts[2,:].argsort()[::-1]]
    pts_x = pts[0, 0:top_k]
    pts_y = pts[1, 0:top_k]

    print(pts)

    plt.imshow(img, cmap='gray')
    plt.scatter(pts_x, pts_y, s=5, color='red')
    plt.axis('off')
    plt.show()
    break

