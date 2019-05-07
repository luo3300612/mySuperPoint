import cv2
from scipy.stats import truncnorm
import numpy as np
import matplotlib.pyplot as plt
import random
import torch


def heatmap2points(heatmap, conf_thresh=0.015, nms_dist=4, border_remove=0):
    H, W = heatmap.shape
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    if len(xs) == 0:
        return np.zeros((3, 0)), None
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    if border_remove != 0:
        bord = border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
    return pts


def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


def wrap_shape(pts, height=480, width=640):
    assert pts.shape == (4, 2)
    pts[:, 0] = pts[:, 0] * width
    pts[:, 1] = pts[:, 1] * height
    return pts


def sample_homography(shape, perspective=True, scaling=True, rotation=True, translation=True,
                      max_sample=5, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
                      perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=np.pi / 2,
                      allow_artifacts=False, translation_overflow=0):
    pts = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

    # center crop
    assert patch_ratio <= 1
    margin = (1 - patch_ratio) / 2
    pts2 = np.array([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]])
    pts2 += margin

    # perspective
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        perspective_displacement = truncnorm.rvs(-perspective_amplitude_y, perspective_amplitude_y)
        h_dislacement_left, h_dislacement_right = truncnorm.rvs(-perspective_amplitude_x, perspective_amplitude_x,
                                                                size=2).tolist()
        pts2 += np.array([[h_dislacement_left, perspective_displacement],
                          [h_dislacement_left, -perspective_displacement],
                          [h_dislacement_right, perspective_displacement],
                          [h_dislacement_right, -perspective_displacement]])

    # scaling
    if scaling:
        center = pts.mean(axis=0)
        count = 0
        pts2_copy = pts2.copy()
        while count < max_sample:
            ratio = truncnorm.rvs(-scaling_amplitude, scaling_amplitude) + 1
            pts2_copy = (pts2 - center) * ratio + center
            if not allow_artifacts:
                if (pts2_copy > 1).any() or (pts2_copy < 0).any():
                    count += 1
                    continue
            # print(ratio)
            pts2 = pts2_copy
            break
        else:
            print(f"sample scaling failed for {max_sample} times")
            print("last pts2:", pts2_copy)
            return None

    # rotation
    if rotation:
        center = pts2.mean(axis=0)
        pts2 = pts2 - center
        pts2_copy = pts2.copy()
        count = 0
        while count < 5 * max_sample:
            angle = np.random.uniform(-max_angle, max_angle)
            M = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            pts2_copy = pts2.dot(M) + center
            # print(angle)
            if not allow_artifacts:
                if (pts2_copy > 1).any() or (pts2_copy < 0).any():
                    count += 1
                    print('invalid rotation')
                    continue

            pts2 = pts2_copy
            break

        else:
            print(f"sample failed for {5 * max_sample} times")
            print("last pts2:", pts2_copy)
            return None

    # translation
    if translation:
        t_min = pts2.min(axis=0)
        t_max = (1 - pts2).min(axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        x, y = np.random.uniform(-t_min, t_max).tolist()
        pts2[:, 0] += x
        pts2[:, 1] += y

    pts = wrap_shape(pts, int(shape[0] * patch_ratio), int(shape[1] * patch_ratio))
    pts2 = wrap_shape(pts2, *shape[:2])
    h, status = cv2.findHomography(pts2, pts)
    return h


def get_heatmap(net, img, cell=8):
    img_tensor = torch.tensor(img)
    img_tensor = img_tensor.view((1, 1, *img.shape)).float()
    output = net(img_tensor)
    output = output.data.cpu().numpy().squeeze()
    # --- Process points.
    dense = np.exp(output)  # Softmax.
    dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    H, W = img.shape[:2]
    Hc = int(H / cell)
    Wc = int(W / cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, cell, cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc * cell, Wc * cell])
    return heatmap


def homographic_adaptation(net, img, Nh, **config):
    sum_heatmap = get_heatmap(net, img)
    count = np.ones(img.shape)
    shape = img.shape
    patch_ratio = 0.5
    for i in range(1, Nh):
        H = sample_homography(shape, patch_ratio=patch_ratio, **config)
        h_img = cv2.warpPerspective(img, H, (
            int(shape[1] * patch_ratio), int(shape[0] * patch_ratio)))  # TODO figure out size to warp
        heatmap = get_heatmap(net, h_img)
        H_inverse = np.linalg.inv(H)
        heatmap_h = cv2.warpPerspective(heatmap, H_inverse, (shape[1], shape[0]))
        mask = (heatmap_h != 0)
        count += mask
        sum_heatmap += heatmap_h
    return sum_heatmap / count


if __name__ == '__main__':
    net = torch.load('/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/result/epoch120',
                     map_location='cpu')
    img = cv2.imread('/run/media/luo3300612/我是D盘~ o(*￣▽￣*)ブ/下载/迅雷下载/coco/train2014/COCO_train2014_000000551710.jpg', 0)

    Nh = 100
    # config = {"allow_artifacts": True, "translation_overflow": 0.2}
    heatmap = homographic_adaptation(net, img, Nh)
    pts = heatmap2points(heatmap, border_remove=0)

    pts_x = pts[0, :]
    pts_y = pts[1, :]
    plt.imshow(img, cmap='gray')
    plt.scatter(pts_x, pts_y, s=5, color='red')
    plt.axis('off')
    plt.show()

    # heatmap = homographic_adaptation(net, img, 10)
    # pts = heatmap2points(heatmap, border_remove=4)
    # pts_x = pts[0, :]
    # pts_y = pts[1, :]
    # plt.imshow(img, cmap='gray')
    # plt.scatter(pts_x, pts_y, s=5, color='red')
    # plt.axis('off')
    # plt.show()
    #
    # heatmap = homographic_adaptation(net, img, 100)
    # pts = heatmap2points(heatmap, border_remove=4)
    # pts_x = pts[0, :]
    # pts_y = pts[1, :]
    # plt.imshow(img, cmap='gray')
    # plt.scatter(pts_x, pts_y, s=5, color='red')
    # plt.axis('off')
    # plt.show()
