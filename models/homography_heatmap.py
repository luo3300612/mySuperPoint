from models.homography import sample_homography as sample_homography
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    height = 480
    width = 640
    Nh = 100
    heatmap = np.ones((height, width))
    fake_img = np.ones((height, width))
    for i in range(1, Nh):
        H = sample_homography((height,width),allow_artifacts=True,translation_overflow=0.2)
        img_h = cv2.warpPerspective(fake_img, H, (320, 240))
        H_invese = np.linalg.inv(H)
        img_origin = cv2.warpPerspective(img_h, H_invese, (width, height))
        heatmap += img_origin
    heatmap = heatmap / np.max(heatmap)
    plt.imshow(heatmap)
    plt.show()
