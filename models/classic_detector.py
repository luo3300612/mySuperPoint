import cv2
import numpy as np
from models.homography import heatmap2points

img = cv2.imread('/run/media/luo3300612/我是D盘~ o(*￣▽￣*)ブ/下载/迅雷下载/coco/train2014/COCO_train2014_000000551710.jpg', 0)

top_k = 500

# img = cv2.resize(img, (320, 240))
out = (np.dstack((img, img, img))).astype('uint8')
# harris
res = cv2.cornerHarris(img, 2, 3, 0.04)

pts = heatmap2points(res, conf_thresh=0.001)

pts = pts[:,pts[2,:].argsort()[::-1]]
print(pts)
pts_x = pts[0, :]
pts_y = pts[1, :]

for i in range(top_k):
    cv2.circle(out, (int(pts_x[i]), int(pts_y[i])), 1, (0, 255, 0), -1, lineType=16)

cv2.imshow("window1", out)
cv2.waitKey(0)


# fast
out = (np.dstack((img, img, img))).astype('uint8')

fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_5_8)
print(fast.getNonmaxSuppression())
kp = fast.detect(img, None)  #
pts = np.empty((3, len(kp)))
for i, point in enumerate(kp):
    pts[0, i], pts[1, i] = point.pt
    pts[2, i] = point.response

pts = pts[:,pts[2,:].argsort()[::-1]]
for i in range(300):
    cv2.circle(out, (int(pts[0, i]), int(pts[1, i])), 1, (0, 255, 0), -1, lineType=16)
cv2.imshow("window", out)
cv2.waitKey(0)

# shi
out = (np.dstack((img, img, img))).astype('uint8')

corners = cv2.goodFeaturesToTrack(img,top_k,0.01,10)#棋盘上的所有点

corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(out, (x,y), 1, (0, 255, 0), -1, lineType=16)

cv2.imshow("window", out)
cv2.waitKey(0)
