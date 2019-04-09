import argparse
import torch
import os
import net_archs
from data_loader.loader import *
import torch.optim as optim
from torch.optim.lr_scheduler import *
from engine import *
from utils.check_point import CheckPoint
from utils.history import History
import numpy as np
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(args):

    set_seed(args.seed)

    # setup logging info
    log_file = '{}/ckpt/{}/{}.log'.format(ROOT_DIR, args.exp, args.ckpt_prefix)
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info(str(args))

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    train_loader, val_loader = ASCDevLoader(device='a').train_val(batch_size=args.batch_size, shuffle=True)

    model = getattr(net_archs, args.net)(args=args).cuda()
    # model = net_archs.CNN3BiGRU(args=args).cuda()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.l2)
    if args.lr_factor < 1.0:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True,
                                      factor=args.lr_factor, patience=args.lr_patience)

    train_hist, val_hist = History(name='train'), History(name='val')

    # checkpoint after new History, order matters
    ckpter = CheckPoint(model=model, optimizer=optimizer, path='{}/ckpt/{}'.format(ROOT_DIR, args.exp),
                        prefix=args.ckpt_prefix, interval=1, save_num=1)

    from utils.utilities import WeightedBCE
    criterion = WeightedBCE(pos_weight=9*torch.ones(10).cuda(), reduction='sum')

    for epoch in range(1, args.run_epochs):
        train_hist.add(
            logs=train_model(train_loader, model, optimizer, criterion, device),
            epoch=epoch
        )
        val_hist.add(
            logs=eval_model(val_loader, model, criterion, device),
            epoch=epoch
        )
        if args.lr_factor < 1.0:
            scheduler.step(val_hist.recent['acc'])

        # plotting
        if args.plot:
            train_hist.clc_plot()
            val_hist.plot()

        # logging
        logging.info("Epoch{:04d},{:6},{}".format(epoch, train_hist.name, str(train_hist.recent)))
        logging.info("Epoch{:04d},{:6},{}".format(epoch, val_hist.name, str(val_hist.recent)))

        ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=val_hist.recent)

    # explicitly save last
    ckpter.save(epoch=args.run_epochs-1, monitor='acc', loss_acc=val_hist.recent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='temp', type=str)
    parser.add_argument('--net', default='CNN_MIL', type=str)
    parser.add_argument('--ckpt_prefix', default='Run01', type=str)
    parser.add_argument('--device', default='5', type=str)
    parser.add_argument('--run_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--pooling', default='max', type=str, choices=['max', 'ave', 'lin', 'exp', 'att'])
    parser.add_argument('--nb_detector', default=4, type=int)
    parser.add_argument('--is_instance_softmax', default=True, type=bool)
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd'])
    parser.add_argument('--l2', default=1e-4, type=float)
    parser.add_argument('--init_lr', default=3e-4, type=float)
    parser.add_argument('--lr_patience', default=3, type=int)
    parser.add_argument('--lr_factor', default=0.5, type=float)
    parser.add_argument('--plot', default=False, type=bool)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('nb_class', default=10, type=int)
    parser.add_argument('combine_type', default='conv1d', choices=['no', 'last', 'conv1d', 'conv2d'])
    args = parser.parse_args()
    run(args)
