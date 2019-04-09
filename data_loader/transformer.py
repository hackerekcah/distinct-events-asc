import torch
# import random
import math
import numpy.random as random
from data_manager.dcase18_taskb import *
from data_manager.taskb_standrizer import *


class ToTensor(object):

    def __call__(self, sample):
        x, y = torch.from_numpy(sample[0]), torch.from_numpy(sample[1])
        x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
        return x, y


if __name__ == '__main__':
    pass
