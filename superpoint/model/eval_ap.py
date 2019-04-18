import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from data import get_test
import numpy as np
from dataset import point2label
from tqdm import tqdm
from tensorboardX import SummaryWriter


def get_heatmap(sample):
    img = sample['img']
    pts = sample['pt']
    img_tensor = torch.tensor(img)
    img_tensor = img_tensor.view((1, 1, *img.shape))
    output = net(img_tensor)
    output = output.squeeze()
    output = np.exp(output.detach().numpy())  # Softmax.
    output = output / (np.sum(output, axis=0) + .00001)  # Should sum to 1.
    output = output[:-1, :, :]
    output = output.transpose(1, 2, 0)
    output = output.reshape((15, 20, 8, 8))
    output = output.transpose(0, 2, 1, 3)
    heatmap = output.reshape(120, 160)
    return heatmap


def mAP_final(test_sample):
    epsilon = 4

    conf_list = []
    label_list = []
    mask_list = []
    index_map_list = []

    for sample in tqdm(test_sample):

        pts = sample['pt'].astype(int)
        label = point2label(pts, binary=True)

        label_mask = label.copy()

        index_map = np.zeros_like(label)
        # check index map code
        #         x = 0
        #         y = 0
        for point in pts:
            x, y = point
            label_mask[max(x - epsilon, 0):min(x + epsilon, label.shape[0]),
            max(y - epsilon, 0):min(y + epsilon, label.shape[1])] = 1
            index_map[max(x - epsilon, 0):min(x + epsilon, label.shape[0]),
            max(y - epsilon, 0):min(y + epsilon, label.shape[1])] = x * 160 + y + len(label_list)

        heatmap = get_heatmap(sample)
        # check index map code
        #         if pts.shape[0] !=0:
        #             try:
        #                 assert heatmap[x,y] == tmp[index_map[x-1,y-1]]
        #                 print("correct")
        #             except AssertionError:
        #                 print(f"{heatmap[x,y]}!={tmp[index_map[x-1,y-1]]},\nwhen index_map[{x-1},{y-1}]=={index_map[x-1,y-1]}")
        #                 print("where",np.where(tmp==heatmap[x,y]))
        #                 raise
        heatmap = list(heatmap.reshape((-1,)))
        label = list(label.reshape((-1,)))
        label_mask = list(label_mask.reshape((-1,)))
        index_map = list(index_map.reshape((-1,)))

        conf_list += heatmap
        label_list += label
        mask_list += label_mask
        index_map_list += index_map

    conf_list = np.array(conf_list)
    label = np.array(label_list)
    label_mask = np.array(mask_list)
    index_map = np.array(index_map_list)

    index = list(reversed(np.argsort(conf_list).tolist()))

    pred_positive_sample_num = 0
    precision_pts = []
    pr_curve = np.zeros((len(index) + 2, 2))

    pr_curve[0, 0] = 0
    pr_curve[0, 1] = 1
    pr_curve[-1, 0] = 1
    pr_curve[-1, 1] = 0

    TP = 0
    positive_sample_num = np.sum(label)
    pred_label = np.zeros_like(conf_list)

    for i in tqdm(range(len(index))):

        ind = index[i]
        #         if label[ind] == 1:
        #             TP += 1
        #         elif  label_mask[ind] == 1:
        #             TP += 1
        #             positive_sample_num += 1

        pred_positive_sample_num += 1

        if label_mask[ind] == 1:
            origin_index = index_map_list[ind]
            if pred_label[origin_index] == 1:
                pred_positive_sample_num -= 1
            else:
                pred_label[origin_index] = 1
                TP += 1

        precision = TP / pred_positive_sample_num
        recall = TP / positive_sample_num

        if recall != 0 and recall != pr_curve[i, 0]:
            precision_pts.append([recall, precision])

        pr_curve[i + 1, 0] = recall
        pr_curve[i + 1, 1] = precision

    return pr_curve, np.array(precision_pts)


if __name__ == '__main__':
    eval_range = range(11, 174)

    writer = SummaryWriter(log_dir="./eval")

    test_data = get_test({"SyntheticData": {"only_point": True}}, loader=False)
    sample_index = np.random.randint(0, 4500, (1000,))
    test_sample = [test_data[i] for i in sample_index]

    for i in tqdm(eval_range):
        net = torch.load(f'/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/result/epoch{i}',
                         map_location='cpu')
        pr_curve,precision_pts = mAP_final(test_sample)
        ap = np.mean(precision_pts[:,1])
        writer.add_scalar("eval_ap/ap",ap,i)
        del pr_curve
        del precision_pts
    writer.close()
