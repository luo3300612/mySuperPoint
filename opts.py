import argparse


def parse_opts():
    parser = argparse.ArgumentParser(description='MagicPoint train parameters')

    parser.add_argument('--img-path', type=str,
                        default='/userhome/mySuperPoint/dataset/data')
    parser.add_argument('--train_info', type=str,
                        default='/userhome/mySuperPoint/dataset/training.csv')
    parser.add_argument('--test_info', type=str,
                        default='/userhome/mySuperPoint/dataset/test.csv')
    parser.add_argument('--val_info', type=str,
                        default='/userhome/mySuperPoint/dataset/validation.csv')
    parser.add_argument('--save_path', type=str,
                        default='/userhome/mySuperPoint/result')
    parser.add_argument('--id', type=str,
                        default='test')

    parser.add_argument('--H', type=int, default=120)
    parser.add_argument('--W', type=int, default=160)
    parser.add_argument('--cell', type=int, default=8)

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--n_iter', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--print_every',type=int,default=10)
    parser.add_argument('--eval_every',type=int,default=2000)
    args = parser.parse_args()
    return args
