from config import Config
from dataset import SyntheticData
from torch.utils.data import DataLoader

dataset_root = Config.dataset_root
train_csv = Config.train_csv
test_csv = Config.test_csv
val_csv = Config.val_csv


def get_train(config, batch_size=64, num_workers=4, loader=True):
    train_data = SyntheticData(train_csv, dataset_root, **config['SyntheticData'])
    if not loader:
        return train_data
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    return train_loader


def get_test(config, batch_size=64, num_worlers=4, loader=True):
    test_data = SyntheticData(test_csv, dataset_root, **config['SyntheticData'])
    if not loader:
        return test_data
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_worlers)
    return test_loader
