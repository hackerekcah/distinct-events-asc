import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
from models.crnn import *
import matplotlib.pyplot as plt
from models.instance_detector import SingleDetector, MultiDetector
from models.milpool import MILPooler


class CNN_MIL(BaseModel):
    """
    """
    def __init__(self, args):
        super(CNN_MIL, self).__init__()

        self.pooling = args.pooling

        # num_features: should be input channel
        self.input_bn = nn.BatchNorm2d(num_features=1)
        self.conv_blocks = ConvBlock(filters=(32, 64, 128))

        self.conv_full_frequency = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 1), padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        self.instance_detector = SingleDetector(feature_dim=256, nb_class=args.nb_class)
        self.pooler = MILPooler(pooling=self.pooling)

    def forward(self, x):
        """
        :param x: (batch, channel, frequency, time) = (batch, 1, 64, 400)
        :return: global_prob: (Batch, Class), frame_prob: (Batch, Time, Class)
        """
        x = self.input_bn(x)                # ->(batch, 1, 64, 400)
        x = self.conv_blocks(x)             # ->(batch, 128, 8, 25)
        x = self.conv_full_frequency(x)     # ->(batch, 256, 1, 25) = (B, C, F, T)

        instance_prob = self.instance_detector(x)
        return self.pooler(instance_prob)


class CNN_MD_MIL(BaseModel):
    """
    """
    def __init__(self, args):
        super(CNN_MD_MIL, self).__init__()

        self.pooling = args.pooling
        self.nb_detector = args.nb_detector
        self.is_instance_softmax = args.is_instance_softmax

        # num_features: should be input channel
        self.input_bn = nn.BatchNorm2d(num_features=1)
        self.conv_blocks = ConvBlock(filters=(32, 64, 128))

        self.conv_full_frequency = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 1), padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        assert self.nb_detector >= 2

        self.instance_detector = MultiDetector(feature_dim=256,
                                               nb_class=args.nb_class,
                                               nb_detector=self.nb_detector,
                                               is_instance_softmax=self.is_instance_softmax)

        self.pooler = MILPooler(pooling=self.pooling)

    def forward(self, x):
        """
        :param x: (batch, channel, frequency, time) = (batch, 1, 64, 400)
        :return: global_prob: (Batch, Class), frame_prob: (Batch, Time, Class)
        """
        x = self.input_bn(x)                # ->(batch, 1, 64, 400)
        x = self.conv_blocks(x)             # ->(batch, 128, 8, 25)
        x = self.conv_full_frequency(x)     # ->(batch, 256, 1, 25) = (B, C, F, T)

        instance_prob = self.instance_detector(x)
        return self.pooler(instance_prob)


class CNN_MTS_MIL(BaseModel):
    """
    Fully Convolutional Model, 4*[(conv-bn-relu)*2-max_pool] + conv256(4,1) + fc(256,10)
    """
    def __init__(self, args):
        super(CNN_MTS_MIL, self).__init__()

        self.pooling = args.pooling

        # num_features: should be input channel
        self.input_bn = nn.BatchNorm2d(num_features=1)
        self.conv_blocks = ConvBlock(filters=(32, 64, 128))

        self.conv_full_frequency = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 1), padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        self.multi_resolution = MultiResolutionBlock(in_channels=256, out_channels=256,
                                                     combine_type=args.combine_type)

        self.instance_detector = SingleDetector(feature_dim=256, nb_class=args.nb_class)

        self.pooler = MILPooler(pooling=self.pooling)

    def forward(self, x):
        """
        :param x: (batch, channel, frequency, time) = (batch, 1, 64, 400)
        :return: global_prob: (Batch, Class), frame_prob: (Batch, Time, Class)
        """
        x = self.input_bn(x)                # ->(batch, 1, 64, 400)
        x = self.conv_blocks(x)             # ->(batch, 128, 8, 25)
        x = self.conv_full_frequency(x)     # ->(batch, 256, 1, 25) = (B, C, F, T)
        x = self.multi_resolution(x)

        instance_prob = self.instance_detector(x)
        return self.pooler(instance_prob)


class CNN_MTS_MD_MIL(BaseModel):
    """
    Fully Convolutional Model, 4*[(conv-bn-relu)*2-max_pool] + conv256(4,1) + fc(256,10)
    """
    def __init__(self, args):
        super(CNN_MTS_MD_MIL, self).__init__()

        self.pooling = args.pooling
        self.nb_detector = args.nb_detector
        self.is_instance_softmax = args.is_instance_softmax

        # num_features: should be input channel
        self.input_bn = nn.BatchNorm2d(num_features=1)
        self.conv_blocks = ConvBlock(filters=(32, 64, 128))

        self.conv_full_frequency = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 1), padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        self.multi_resolution = MultiResolutionBlock(in_channels=256, out_channels=256,
                                                     combine_type=args.combine_type)
        assert self.nb_detector >= 2
        self.instance_detector = MultiDetector(feature_dim=256,
                                               nb_class=args.nb_class,
                                               nb_detector=self.nb_detector,
                                               is_instance_softmax=self.is_instance_softmax)
        self.pooler = MILPooler(pooling=self.pooling)

    def forward(self, x):
        """
        :param x: (batch, channel, frequency, time) = (batch, 1, 64, 400)
        :return: global_prob: (Batch, Class), frame_prob: (Batch, Time, Class)
        """
        x = self.input_bn(x)                # ->(batch, 1, 64, 400)
        x = self.conv_blocks(x)             # ->(batch, 128, 8, 25)
        x = self.conv_full_frequency(x)     # ->(batch, 256, 1, 25) = (B, C, F, T)
        x = self.multi_resolution(x)

        instance_prob = self.instance_detector(x)
        return self.pooler(instance_prob)


if __name__ == '__main__':

    pass
