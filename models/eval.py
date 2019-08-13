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
from tensorboardX import SummaryWriter


# epoch17 nms_dist=4,conf_thresh=0.001,border_remove=0 ===> p=0.6584 r = 0.6922
# epoch17 nms_dist=4,conf_thresh=1/65,border_remove=0 ===> p=0.7251 r = 0.7065
# epoch 40 ===> p 0.7479 r 0.7452
# epoch 72 ===> p 0.7959 r 0.789
# epoch 74 ===> p 0.7981 r 0.7869
# epoch 80 ===> p 0.7995 r 0.7865
# epoch 95 ===> p 0.8039 r 0.7901 *
# epoch 120 ===> p 0.8071 r 0.7923
# numworker=1 144s
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

        pred_positive_num = np.sum(pred_label)
        positive_num = np.sum(ground_truth_label)

        if not pred_positive_num or not positive_num:  # equal zero
            if pred_positive_num == positive_num:
                precision_sum += 1
                recall_sum += 1
            elif pred_positive_num == 0:
                precision_sum += 1
                recall_sum += 0
            elif positive_num == 0:
                precision_sum += 0
                recall_sum += 1

        else:
            precision_sum += np.sum(np.logical_and(pred_label, ground_truth_label)) / np.sum(pred_label == 1)
            recall_sum += np.sum(
                np.logical_and(pred_label, ground_truth_label)) / np.sum(
                ground_truth_label == 1)
    ed = time.time()
    print(f'Task {name} runs {ed - st:.2f} seconds.')
    return precision_sum, recall_sum


if __name__ == '__main__':

    to_eval = range(108, 133)
    writer = SummaryWriter(log_dir="./eval")
    num_worker = 1

    for model_i in to_eval:
        print('Parent process %s.' % os.getpid())

        fe = SuperPointFrontend(
            weights_path=f'/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/result/epoch{model_i + 1}',
            nms_dist=4,
            conf_thresh=1 / 65,
            border_remove=0)

        config = {"SyntheticData": {"only_point": True}}
        test_data = get_test(config, loader=False)

        st = datetime.now()
        if num_worker > 1:  # use multiprocess
            interval = np.linspace(0, len(test_data), num_worker + 1, endpoint=True, dtype=int)

            p = Pool(num_worker)
            results = [p.apply_async(eval, args=(i, interval[i], interval[i + 1])) for i in range(num_worker)]
            print('Waiting for all subprocesses done...')
            p.close()
            p.join()
            print('All subprocesses done.')

            avg_precision = sum([item.get()[0] for item in results]) / len(test_data)
            avg_recall = sum([item.get()[1] for item in results]) / len(test_data)
        else:
            precision_sum, recall_sum = eval(1, 0, len(test_data))
            avg_precision = precision_sum / len(test_data)
            avg_recall = recall_sum / len(test_data)

        ed = datetime.now()
        timedelta = (ed - st).seconds

        print(f"avg_precision:{avg_precision:.4f}")
        print(f"avg_recall:{avg_recall:.4f}")
        print(f"{timedelta} seconds")
        writer.add_scalar("eval3/precision", avg_precision, model_i)
        writer.add_scalar("eval3/recall", avg_recall, model_i)
        writer.add_scalar("eval3/f1", 2 * (avg_recall * avg_precision) / (avg_recall + avg_precision), model_i)

    writer.close()
