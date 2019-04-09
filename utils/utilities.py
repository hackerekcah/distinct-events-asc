import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch


def calculate_confusion_matrix(target, predict, classes_num):
    """Calculate confusion matrix.
    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes
    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """

    confusion_matrix = np.zeros((classes_num, classes_num))
    samples_num = len(target)

    for n in range(samples_num):
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, title, labels, values):
    """Plot confusion matrix.
    Inputs:
      confusion_matrix: matrix, (classes_num, classes_num)
      labels: list of labels
      values: list of values to be shown in diagonal
    Ouputs:
      None
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    if labels:
        ax.set_xticklabels([''] + labels, rotation=90, ha='left')
        ax.set_yticklabels([''] + labels)
        ax.xaxis.set_ticks_position('bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    for n in range(len(values)):
        plt.text(n - 0.4, n, '{:.2f}'.format(values[n]), color='yellow')

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.tight_layout()
    plt.show()


def calculate_accuracy(target, predict, classes_num, average=None):
    """Calculate accuracy.
    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
    Outputs:
      accuracy: float
    """

    samples_num = len(target)

    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')


def weighted_binary_cross_entropy(output, target, pos_weight=None, reduction='sum'):
    """
    :param output: prediction probabilities
    :param target:
    :param pos_weight: tensor with len same to number of class
    :param reduction:
    :return:
    """

    EPS = 1e-12

    if pos_weight is not None:
        assert len(pos_weight) == target.size(1)

        loss = pos_weight * (target * torch.log(output + EPS)) + ((1 - target) * torch.log(1 - output + EPS))
    else:
        loss = target * torch.log(output + EPS) + (1 - target) * torch.log(1 - output + EPS)

    if reduction == 'sum':
        return torch.neg(torch.sum(loss))
    elif reduction == 'mean':
        return torch.neg(torch.mean(loss))


class WeightedBCE:
    def __init__(self, pos_weight=None, reduction='sum'):
        self.pos_weight = pos_weight
        self.reduction = reduction

    def __call__(self, output, target):
        return weighted_binary_cross_entropy(output, target, pos_weight=self.pos_weight, reduction=self.reduction)


class OrderedWeightedAverageRankLoss:
    def __init__(self, reduction='sum', is_order_weighted=False, margin=1.0):
        """
        see paper 'ICLR,09, Ranking with ordered weighted pairwise classification'

        :param reduction:
        :param is_order_weighted:
            if True, use ordered weight
            if False, use mean weight, then this is just a multi-class margin loss / hinge loss
        """
        self.reduction = reduction
        self.is_order_weighted = is_order_weighted
        self.margin = margin

    def __call__(self, output, target):
        """

        :param output: rank score, usually net logits for each class, (Batch, nb_class)
        :param target: one-hot label, (Batch, nb_class)
        :return:
        """

        # (Batch, 1), score of the true label
        rank_score_t = torch.sum(output * target, dim=1, keepdim=True)
        # (Batch, nb_class), remove output[true_label] from loss
        margin_loss = torch.nn.functional.relu(output - rank_score_t + self.margin) * (1 - target)

        if self.is_order_weighted:
            # sort loss
            loss_sorted, _ = torch.sort(margin_loss, dim=1, descending=True)
            # count nonzeros for each row, (batch,)
            nb_vialation = loss_sorted.size(1) - (loss_sorted == 0).sum(1)

            weights = self.order_weight(nb_vialation=nb_vialation, nb_class=target.size(1))

            weights = weights.to('cuda')

            # (batch,)
            weighted_loss = torch.sum(loss_sorted * weights, dim=1)

        else:
            # (batch,)
            weighted_loss = torch.mean(margin_loss, dim=1)

        if self.reduction == 'sum':
            loss = torch.sum(weighted_loss)
        else:
            loss = torch.mean(weighted_loss)

        return loss

    def order_weight(self, nb_vialation, nb_class):

        with torch.no_grad():

            weights = []

            # for each element in nb_vialation
            for m in nb_vialation:
                m = m.type(torch.long)
                if m > 0:
                    alpha = torch.zeros(nb_class)
                    alpha[:m] = 1 / torch.arange(start=1, end=m+1, dtype=torch.float)
                    # normalize to have sum=1
                    alpha = alpha / torch.sum(alpha)
                    weights.append(alpha.view(1, -1))
                else:
                    weights.append(torch.zeros(1, nb_class))

            # (batch, nb_class)
            weights = torch.cat(weights, dim=0)

            return weights






