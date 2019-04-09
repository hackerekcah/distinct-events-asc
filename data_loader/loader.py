from torch.utils.data import Dataset, DataLoader
from data_loader.data_sets import *
from data_loader.transformer import *
from torchvision import transforms
import torch


class ASCDevLoader:
    """
    Dcase18
    """
    def __init__(self, device='a'):
        self.device = device
        self.train_set = TaskbDevSet(mode='train', device=device, transform=ToTensor())
        self.val_set = TaskbDevSet(mode='test', device=device, transform=ToTensor())

    def train_val(self, batch_size=128, shuffle=True):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle)

        # Not need to shuffle validation data
        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader


class Dcase17DevLoader:
    """
    Dcase17
    """
    def __init__(self, fold_idx=1):
        self.fold_idx = fold_idx
        self.train_set = Dcase17DevSet(mode='train', fold_idx=fold_idx, transform=ToTensor())
        self.val_set = Dcase17DevSet(mode='test', fold_idx=fold_idx, transform=ToTensor())

    def train_val(self, batch_size=128, shuffle=True):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle)

        # Not need to shuffle validation data
        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader


class Dcase17EvaLoader:
    """
    Dcase17
    """
    def __init__(self):
        self.dev_set = Dcase17EvaSet(split='dev', transform=ToTensor())
        self.eva_set = Dcase17EvaSet(split='eva', transform=ToTensor())

    def dev_eva(self, batch_size=128, shuffle=True):
        dev_loader = DataLoader(dataset=self.dev_set, batch_size=batch_size, shuffle=shuffle)

        # Not need to shuffle validation data
        eva_loader = DataLoader(dataset=self.eva_set, batch_size=batch_size, shuffle=False)
        return dev_loader, eva_loader

