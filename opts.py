import argparse


def parse_opts():
    parser = argparse.ArgumentParser(description='MagicPoint train parameters')

    parser.add_argument('--img-path', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/dataset/data')
    parser.add_argument('--train-info', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/model/training.csv')
    parser.add_argument('--test-info', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/model/test.csv')
    parser.add_argument('--val-info', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/mySuperPoint/superpoint/model/validation.csv')
    parser.add_argument('--save-path', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/mySuperPoint/result')
    parser.add_argument('--id', type=str,
                        default='test')

    parser.add_argument('--H', type=int, default=120)
    parser.add_argument('--W', type=int, default=160)
    parser.add_argument('--cell', type=int, default=8)

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--n_epoch', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers',type=int,default=4)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    return args