import torch


def train_model(train_loader, model, optimizer, criterion, device):
    """
    Note: train_loss and train_acc is accurate only if set drop_last=False in loader

    :param train_loader: y: one_hot float tensor
    :param model:
    :param optimizer:
    :param criterion: set reduction='sum'
    :param device:
    :return:
    """
    model.train(mode=True)
    train_loss = 0
    correct = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        global_prob = model(x)[0]
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            _, yi = y.max(dim=1)
            loss = criterion(global_prob, yi)
        else:
            loss = criterion(global_prob, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        with torch.no_grad():
            pred = global_prob.max(1, keepdim=True)[1]  # get the index of the max log-probability
            _, y_idx = y.max(dim=1)
            correct += pred.eq(y_idx.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    return {'loss': train_loss, 'acc': train_acc}


def eval_model(test_loader, model, criterion, device):

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            global_prob = model(data)[0]
            # to make BCELoss stable, avoid log(0)
            # global_prob.clamp_(min=1e-7, max=1 - 1e-7)
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                _, yi = target.max(dim=1)
                loss = criterion(global_prob, yi)
            else:
                loss = criterion(global_prob, target)
            test_loss += loss.item()
            # get the index of the max log-probability
            pred = global_prob.max(1, keepdim=True)[1]
            _, target_idx = target.max(dim=1)
            correct += pred.eq(target_idx.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    return {'loss': test_loss, 'acc': test_acc}
