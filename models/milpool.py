import torch
import torch.nn as nn
import torch.nn.functional as F


class MILPooler(nn.Module):
    def __init__(self, pooling, feature_dim=None, nb_class=None):
        super(MILPooler, self).__init__()

        self.pooling = pooling
        self.feature_dim = feature_dim
        self.nb_class = nb_class

        if self.pooling == 'att':
            # weight is shared among each instance
            assert self.feature_dim and self.nb_class
            self.fc_att = nn.Linear(feature_dim, nb_class)
            # better initialization
            nn.init.xavier_uniform_(self.fc_att.weight)
            nn.init.constant_(self.fc_att.bias, 0)

    def forward(self, instance_prob, x=None):
        """
        :param instance_prob:
        :param x: should specify x if self.pooling='att', x.shape=(B, C, 1, T)
        :return:
        """
        if self.pooling == 'max':
            bag_prob, _ = instance_prob.max(dim=1)
            return bag_prob, instance_prob
        elif self.pooling == 'ave':
            bag_prob = instance_prob.mean(dim=1)
            return bag_prob, instance_prob
        elif self.pooling == 'lin':
            bag_prob = (instance_prob * instance_prob).sum(dim=1) / instance_prob.sum(dim=1)
            return bag_prob, instance_prob
        elif self.pooling == 'exp':
            bag_prob = (instance_prob * instance_prob.exp()).sum(dim=1) / instance_prob.exp().sum(dim=1)
            return bag_prob, instance_prob
        elif self.pooling == 'att':
            x = x.view(x.size(0), x.size(1), int(x.size(2) * x.size(3)))    # (B, C, F, T) -> (B, C, F*T)
            x = torch.permute(0, 2, 1)                                      # (B, C, F*T)   -> (B, F*T/nb_ins, C)
            instance_att = F.softmax(self.fc_att(x), dim=1)    # (Batch, nb_ins, feature_dim) -> (B, nb_ins, nb_class)
            bag_prob = (instance_prob * instance_att).sum(dim=1)
            return bag_prob, instance_prob, instance_att
