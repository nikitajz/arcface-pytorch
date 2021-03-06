from __future__ import print_function

import json
import os
from collections import OrderedDict

import torch
from adabound import AdaBound
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils import data

import environments
from callback import TensorboardLogger, LoggingCallback, CallbackManager, SlackNotifyCallback, WeightCheckpointCallback
from config import Config
from data.dataset import get_dataset
from evaluation import LFWEvaluation, CFPEvaluation
from models.focal_loss import FocalLoss
from models.metrics import AddMarginProduct, ArcMarginProduct, SphereProduct
from models.resnet import get_model
from utils.logger import get_logger
from utils.serializer import class_to_dict

logger = get_logger(__name__, output_file=os.path.join(Config.checkpoints_path, 'log.txt'))


def top_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            x = correct_k.mul_(100.0 / batch_size)
            x = x.data.cpu().numpy()[0]
            res.append(x)
    return res


def save_model(model, save_path, name, iter_cnt):
    os.makedirs(save_path, exist_ok=True)

    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    logger.info(f'save {name} to {save_name}')
    return save_name


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def calculate_metrics(output, label) -> dict:
    y_pred = output.data.cpu().numpy()
    y_true = label.data.cpu().numpy()

    top_k = [1, 3, 5, 10]
    acc = top_accuracy(output, label, topk=top_k)

    with torch.no_grad():
        logloss = torch.nn.CrossEntropyLoss()(output, label)

    data = OrderedDict()
    data['logloss'] = logloss.item()

    for k, acc in zip(top_k, acc):
        data[f'acc@{k}'] = acc
    return data


class LinearMetrics(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(LinearMetrics, self).__init__()
        self.linear = nn.Linear(input_dim, out_dim)

    def forward(self, x, label):
        return self.linear(x)


if __name__ == '__main__':

    opt = Config()
    device = torch.device("cuda")

    train_dataset = get_dataset(Config.dataset, phase='train', input_shape=environments.INPUT_SHAPE)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=opt.train_batch_size,
                                   shuffle=True,
                                   num_workers=opt.num_workers)

    logger.info('{} train iters per epoch:'.format(len(train_loader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    elif opt.loss == 'logloss':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError()

    model = get_model(opt.backbone)()

    if Config.pretrained_model_path:
        logger.info(f'load weight from {Config.pretrained_model_path}')
        model.load_state_dict(torch.load(Config.pretrained_model_path))

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, train_dataset.n_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, train_dataset.n_classes, s=30, m=0.35, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, train_dataset.n_classes, m=4)
    elif opt.metric == 'linear':
        metric_fc = LinearMetrics(512, train_dataset.n_classes)
    else:
        raise ValueError('Invalid Metric Name: {}'.format(opt.metric))

    if Config.pretrained_metric_path:
        logger.info(f'load metric weight from {Config.pretrained_metric_path}')
        metric_fc.load_state_dict(torch.load(Config.pretrained_metric_path))

    logger.info(model)
    model.to(device)
    metric_fc.to(device)

    params = [{'params': model.parameters()}, {'params': metric_fc.parameters()}]
    if Config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=opt.lr, weight_decay=opt.weight_decay, momentum=.9,
                                    nesterov=True)
    elif Config.optimizer == 'adabound':
        optimizer = AdaBound(params=params,
                             lr=opt.lr,
                             final_lr=opt.final_lr,
                             amsbound=opt.amsbound)
    elif Config.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        raise ValueError('Invalid Optimizer Name: {}'.format(Config.optimizer))
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    callback_manager = CallbackManager([
        TensorboardLogger(log_dir=Config.checkpoints_path),
        LoggingCallback(),
        WeightCheckpointCallback(save_to=Config.checkpoints_path,
                                 metric_model=metric_fc)
    ])

    eval_functions = [
        CFPEvaluation(eval_type='ff'),
        CFPEvaluation(eval_type='fp'),
        LFWEvaluation()
    ]

    if environments.SLACK_INCOMMING_URL and not Config.is_debug:
        logger.info('Add Slack Notification')
        callback_manager.callbacks.append(SlackNotifyCallback(url=environments.SLACK_INCOMMING_URL, config=Config))

    with open(Config.config_path, 'w') as f:
        data = class_to_dict(Config)
        json.dump(data, f, indent=4, sort_keys=True)
        logger.info(f'save config to {Config.config_path}')

    try:
        for epoch in range(opt.max_epoch):
            scheduler.step()
            model.train()
            callback_manager.on_epoch_start(epoch)

            for i, data in enumerate(train_loader):
                callback_manager.on_batch_start(n_batch=i)
                data_input, label = data
                data_input = data_input.to(device)
                label = label.to(device).long()
                feature = model(data_input)
                output = metric_fc(feature, label)
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iters = epoch * len(train_loader) + i

                metric = calculate_metrics(output, label)
                metric[Config.loss] = loss.item()
                metric['lr'] = get_lr(optimizer)
                callback_manager.on_batch_end(loss=loss.item(), n_batch=i, train_metric=metric)
                if Config.is_debug:
                    break
            if epoch % opt.save_interval == 0 or epoch == opt.max_epoch:
                save_model(model, opt.checkpoints_path, opt.backbone, epoch)
                save_model(metric_fc, opt.checkpoints_path, opt.metric, epoch)

            # eval on LDW Dataset
            model.eval()
            valid_metrics = dict()
            for f in eval_functions:
                val_data = f.call(model, input_shape=environments.INPUT_SHAPE, device=device)
                valid_metrics.update(val_data)

            callback_manager.on_epoch_end(epoch, valid_metric=valid_metrics)

        callback_manager.on_end_train()
    except KeyboardInterrupt as e:
        callback_manager.on_end_train(e)

    except Exception as e:
        import traceback

        logger.warning(traceback.format_exc())
        callback_manager.on_end_train(e)
