import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter()
for epoch in range(100):
    writer.add_scalar('scalar/test', np.random.rand(), epoch)
    writer.add_scalars('scalar/scalars_test', {'xsinx': epoch * np.sin(epoch), 'xcosx': epoch * np.cos(epoch)}, epoch)

writer.close()


# import cv2
#
# img = cv2.imread('/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/dataset/data/draw_checkerboard/images/test/5.png',0)
# print(img.shape)


import numpy as np

a = np.array