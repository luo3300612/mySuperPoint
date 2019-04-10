import numpy as np
from config import Config

H = Config.H
W = Config.W
Hc = Config.Hc
Wc = Config.Wc


def output2points(output, alpha=0.001):
    output = np.exp(output.detach().numpy())  # Softmax.
    output = output / (np.sum(output, axis=0) + .00001)  # Should sum to 1.
    output = output[:-1, :, :]
    output = output.transpose(1, 2, 0)
    output = output.reshape((Hc, Wc, 8, 8))
    output = output.transpose(0, 2, 1, 3)
    output = output.reshape(H, W)
    # print(output.shape)
    # print(output[output > alpha])
    xs, ys = np.where(output > alpha)

    points = np.vstack((xs, ys)).T
    # if len(points) == 0:
    # print(f'there are {len(points)} points')
    # print(f"max output value is {np.max(output)}")
    # print(f"min output value is {np.min(output)}")
    return points
