import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDetector(nn.Module):
    """
    inspired by sub_concept layer as in "DeepMIML" paper
    """
    def __init__(self, feature_dim, nb_class, nb_detector=4, is_instance_softmax=True):
        super(MultiDetector, self).__init__()

        self.nb_detector = nb_detector
        self.nb_class = nb_class
        self.is_instance_softmax = is_instance_softmax

        self.detector_mapping = nn.Conv2d(in_channels=feature_dim,
                                          out_channels=self.nb_detector * self.nb_class,
                                          kernel_size=(1, 1))
        nn.init.xavier_uniform_(self.detector_mapping.weight)
        nn.init.constant_(self.detector_mapping.bias, 0)

    def forward(self, x):
        """
        :param x: (B, C, 1, T)
        :return: instance_prob of shape (B, T, L)
        """

        x = self.detector_mapping(x)                                  # ->(B, L*K, 1, T)
        x = x.view(x.size(0),
                   self.nb_class,
                   self.nb_detector,
                   x.size(-1))                                          # -> (B, L, K, T)
        # pool over sub-concept
        x = F.max_pool2d(x, kernel_size=(self.nb_detector, 1))           # -> (B, L, 1, T)
        x = torch.squeeze(x, dim=-2)                                    # -> (B, L, T)
        x = x.permute(0, 2, 1)                                          # -> (B, T, L)

        if self.is_instance_softmax:
            instance_prob = F.softmax(x, dim=-1)                        # -> (B, T, L) softmax-ed
        else:
            instance_prob = torch.sigmoid(x)

        return instance_prob


class SingleDetector(nn.Module):
    """
    A linear classifier, one detector for each scene
    """
    def __init__(self, feature_dim, nb_class):
        """
        :param feature_dim:
        :param nb_class:
        """
        super(SingleDetector, self).__init__()

        self.conv = nn.Conv2d(in_channels=feature_dim, out_channels=nb_class, kernel_size=(1, 1))
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):

        x = self.conv(x)                # (B, feature_dim, F, T) -> (B, nb_class, F, T)
        x = torch.squeeze(x, dim=2)     # (B, nb_class, F=1, T) -> (B, nb_class, T)
        instance_logits = x.permute(0, 2, 1)  # (B, nb_class, T) -> (B, T, nb_class)

        instance_prob = torch.sigmoid(instance_logits)
        return instance_prob
