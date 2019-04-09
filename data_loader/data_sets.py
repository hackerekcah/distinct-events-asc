from torch.utils.data import Dataset
from data_manager.dcase18_taskb import Dcase18TaskbData
from data_manager.taskb_standrizer import TaskbStandarizer
import numpy as np


class TaskbDevSet (Dataset):
    def __init__(self, mode='train', device='a', norm_device=None, transform=None):
        super(TaskbDevSet, self).__init__()

        if not norm_device:
            norm_device = device

        # x.shape(Bath, Hight, Width)
        self.x, self.y = TaskbStandarizer(data_manager=Dcase18TaskbData()).\
            load_dev_standrized(mode=mode, device=device, norm_device=norm_device)

        self.x = np.expand_dims(self.x, axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class Dcase17DevSet (Dataset):
    def __init__(self, mode='train', fold_idx=1, transform=None):
        super(Dcase17DevSet, self).__init__()

        # x.shape(Bath, Hight, Width)
        from data_manager.dcase17_manager import Dcase17Data
        from data_manager.dcase17_stdrizer import Dcase17Standarizer
        self.x, self.y = Dcase17Standarizer(data_manager=Dcase17Data()).\
            load_dev_standrized(mode=mode, fold_idx=fold_idx)

        self.x = np.expand_dims(self.x, axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class Dcase17EvaSet (Dataset):
    def __init__(self, split='dev', transform=None):
        super(Dcase17EvaSet, self).__init__()

        # x.shape(Bath, Hight, Width)
        from data_manager.dcase17_manager import Dcase17Data
        from data_manager.dcase17_stdrizer import Dcase17Standarizer
        self.x, self.y, _= Dcase17Standarizer(data_manager=Dcase17Data()).\
            load_eva_fname_standrized(split=split)

        self.x = np.expand_dims(self.x, axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample
