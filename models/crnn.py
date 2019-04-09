import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import matplotlib.pyplot as plt


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        pass

    def predict(self, x, verbose=True, batch_size=100):
        """
        Predict in batches. Both input and output are numpy arrays.
        :param x:
        :param verbose:
            If verbose == True, return all of global_prob, frame_prob and att
            If verbose == False, only return global_prob
        :param batch_size:
        :return: tuple of (bag preds, instance preds, attention weight), last two items may not exist
        """
        self.eval()
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                x_tensor = Variable(torch.from_numpy(x[i: i + batch_size])).cuda().type(torch.float)
                output = self.forward(x_tensor)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]

    def plot(self, x, wav_name=None):
        """
        plot spectrogram, instance prediction, bag prediction
        :param x: (1, 1, F, T), numpy
        :param wav_name: plot title
        :return:
        """
        plt.figure(dpi=150)
        plt.title(wav_name) if wav_name else None
        # plot spectrogram
        plt.imshow(numpy.squeeze(x), origin='lower')  # (F, T)

        # get prediction result
        result = self.predict(x)
        instant_pred = numpy.squeeze(result[1])                # (1, T, C) -> (T, C)
        instant_pred = numpy.repeat(instant_pred, int(x.shape[-1] / instant_pred.shape[0]), axis=0)
        bag_pred = numpy.squeeze(result[0])                    # shape (1, C)
        max_instance = instant_pred[:, bag_pred.argmax(axis=0)] * x.shape[-2]
        plt.plot(max_instance, color='orange')

        # plot instance pred
        plt.figure(dpi=150)
        plt.title(wav_name) if wav_name else None
        lines = ["-", "--", "-.", ':', (0, (1, 5))]
        from itertools import cycle
        linecycler = cycle(lines)
        for i in range(instant_pred.shape[1]):
            plt.plot(instant_pred[:, i], linestyle=next(linecycler), linewidth=2)

        nb_class = instant_pred.shape[1]
        if nb_class == 10:

            legend = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
                      'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']
        elif nb_class == 15:
            legend = ['beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
                      'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
                      'office', 'park', 'residential_area', 'train', 'tram']
        legend_score = [l + "({:.2f})".format(bag_pred[i]) for i, l in enumerate(legend)]
        plt.legend(legend_score, bbox_to_anchor=(1, 1))

        # plot bag
        plt.figure(dpi=150)
        plt.title(wav_name) if wav_name else None
        plt.plot(bag_pred)
        plt.xticks(ticks=numpy.arange(nb_class), labels=legend, rotation=60)
        plt.show()


class ConvBlock(nn.Module):
    def __init__(self, filters=(16, 32, 64, 128)):
        super(ConvBlock, self).__init__()

        self.nb_blocks = len(filters)
        self.filters = list(filters)
        self.filters.insert(0, 1)
        self.conv_blocks = nn.ModuleList()
        for i in range(self.nb_blocks):
            self.conv_blocks.append(
                self._conv_bn_relu_x2_mp(in_channels=self.filters[i], out_channels=self.filters[i+1])
            )

    def _conv_bn_relu_x2_mp(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=0)
        )

    def forward(self, x):
        for i in range(self.nb_blocks):
            x = self.conv_blocks[i](x)
        return x


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


class MultiResolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, combine_type='no'):
        """

        :param in_channels:
        :param out_channels:
        :param combine_type:
            no
            conv2d
            conv1d
            last
        """
        super(MultiResolutionBlock, self).__init__()
        # (B, C, T) -> (B, C, T)
        self.dilated_conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=3, dilation=1, padding=1, stride=1)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)

        self.dilated_conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                       kernel_size=3, dilation=2, padding=2, stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.dilated_conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=3, dilation=4, padding=4, stride=1)
        self.bn3 = nn.BatchNorm1d(num_features=out_channels)

        self.combine_type = combine_type

        if combine_type == 'conv1d':

            self.combine = nn.Conv1d(in_channels=4*out_channels, out_channels=out_channels, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm1d(num_features=out_channels)
        elif combine_type == 'conv2d':
            self.combine = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1, 1), bias=False)
            self.bn4 = nn.BatchNorm2d(num_features=1)

    def forward(self, x):
        """
        :param x: (B, C, F, T), usually F=1
        :return:
        """
        # (B, C, F, T)->(B, C*F, T) = (B, nb_feature, T)
        x1 = x.view(x.size(0), x.size(1) * x.size(2), x.size(-1))
        # (B, nb_feature, T), time resolution x 2
        x2 = F.relu(self.bn1(self.dilated_conv1(x1)))
        # (B, nb_feature, T), time resolution x 2
        x3 = F.relu(self.bn2(self.dilated_conv2(x2)))
        # (B, nb_feature, T)
        x4 = F.relu(self.bn3(self.dilated_conv3(x3)))

        if self.combine_type == 'conv1d':

            # (B, 4*nb_feature, T)
            x5 = torch.cat([x1, x2, x3, x4], dim=1)

            # (B, nb_feature, T)
            x6 = F.relu(self.bn4(self.combine(x5)))
            out = x6.view(x6.size(0), x6.size(1), 1, x6.size(-1))   # (B, nb_feature, 1, T)

        # (B, 4, nb_feature, T)
        elif self.combine_type == 'conv2d':

            # add axis at dim1 and concat, # (B, 4, nb_feature, T)
            fmaps = torch.stack([x1, x2, x3, x4], dim=1)

            # (B, 1, nb_feature, T)
            x5 = F.relu(self.bn4(self.combine(fmaps)))

            # (B, nb_feature, 1, T)
            out = x5.view(x5.size(0), x5.size(2), 1, x5.size(-1))

        elif self.combine_type == 'last':
            # (B, nb_feature, T) -> # (B, nb_feature, 1, T)
            out = x4.view(x4.size(0), x4.size(1), 1, x4.size(-1))

        elif self.combine_type == 'no':
            # concat over time axis, ->(B, nb_features, T*4)
            out = torch.cat([x1, x2, x3, x4], dim=2)

            # (B, nb_features, 1, T * 3)
            out = out.view(out.size(0), out.size(1), 1, out.size(2))

        return out



