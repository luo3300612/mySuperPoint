import torch
from utils import SuperPointFrontend
from config import Config
from torch.utils.data import DataLoader
from data import get_test
from dataset import point2label
import numpy as np
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool
import os, time, random


# epoch17 nms_dist=4,conf_thresh=0.001,border_remove=0 ===> p=0.6584 r = 0.6922
# epoch17 nms_dist=4,conf_thresh=1/65,border_remove=0 ===> p=0.7251 r = 0.7065 144s
# epoch 40 ===> p 0.7479 r 0.7452
# epoch 80 ===> p 0.7995 r 0.7865
# numworker=2 105s


def eval(name, start, end):
    print('Run task %s (%s)...' % (name, os.getpid()))
    st = time.time()
    precision_sum = 0
    recall_sum = 0

    for i in tqdm(range(start, end), desc=str(name), position=name, leave=False):
        sample = test_data[i]
        img = sample['img']
        pts = sample['pt']
        ground_truth_label = point2label(pts, binary=True)
        pred_pts_and_conf, heatmap = fe.run(img)
        pred_pts = pred_pts_and_conf[[1, 0], :].T
        pred_label = point2label(pred_pts, binary=True)
        if not np.sum(pred_label == 1) or not np.sum(ground_truth_label == 1):
            if np.sum(pred_label == 1) == np.sum(ground_truth_label == 1):
                precision_sum += 1
                recall_sum += 1
            else:
                precision_sum += 0
                recall_sum += 0
        else:
            precision_sum += np.sum(np.logical_and(pred_label, ground_truth_label)) / np.sum(pred_label == 1)
            recall_sum += np.sum(
                np.logical_and(pred_label, ground_truth_label)) / np.sum(
                ground_truth_label == 1)
    ed = time.time()
    print(f'Task {name} runs {ed - st:.2f} seconds.')
    return precision_sum, recall_sum


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())

    fe = SuperPointFrontend(weights_path='/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/result/epoch92',
                            nms_dist=4,
                            conf_thresh=1 / 65,
                            border_remove=0)

    config = {"SyntheticData": {"only_point": True}}

    test_data = get_test(config, loader=False)

    num_worker = 2
    interval = np.linspace(0, len(test_data), num_worker + 1, endpoint=True, dtype=int)

    st = datetime.now()
    p = Pool(num_worker)
    results = [p.apply_async(eval, args=(i, interval[i], interval[i + 1])) for i in range(num_worker)]
    # precision_sum += result[0]
    # recall_sum += result[1]
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

    avg_precision = sum([item.get()[0] for item in results]) / len(test_data)
    avg_recall = sum([item.get()[1] for item in results]) / len(test_data)

    ed = datetime.now()
    timedelta = (ed - st).seconds

    print(f"avg_precision:{avg_precision:.4f}")
    print(f"avg_recall:{avg_recall:.4f}")
    print(f"{timedelta} seconds")