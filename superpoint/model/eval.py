import torch
from utils import SuperPointFrontend
from config import Config
from torch.utils.data import DataLoader
from data import get_test
from dataset import point2label
import numpy as np
from datetime import datetime
from tqdm import tqdm

# epoch17 nms_dist=4,conf_thresh=0.001,border_remove=0 ===> p=0.6584 r = 0.6922
# epoch17 nms_dist=4,conf_thresh=1/65,border_remove=0 ===> p=0.7251 r = 0.7065
fe = SuperPointFrontend(weights_path='/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/result/epoch17',
                        nms_dist=4,
                        conf_thresh=1/65,
                        border_remove=0)

config = {"SyntheticData": {"only_point": True}}

test_data = get_test(config, loader=False)

precision_sum = 0
recall_sum = 0

start = datetime.now()

for sample in tqdm(test_data):
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

end = datetime.now()
timedelta = (end - start).seconds
avg_precision = precision_sum / len(test_data)
avg_recall = recall_sum / len(test_data)

print(f"avg_precision:{avg_precision:.4f}")
print(f"avg_recall:{avg_recall:.4f}")
print(f"{timedelta} seconds pre img")
