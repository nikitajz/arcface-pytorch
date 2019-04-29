from __future__ import print_function

from datetime import datetime

from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from torch.utils import data

from callbacks import TensorboardLogger, LoggingCallback, Callbacks
from data.dataset import CASIADataset
from test import *
from utils import Visualizer
from utils.logger import get_logger

logger = get_logger(__name__)


def save_model(model, save_path, name, iter_cnt):
    os.makedirs(save_path, exist_ok=True)

    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def get_time_string(t=None, formatting='{0:%Y-%m-%d_%H-%M-%S}'):
    if t is None:
        t = datetime.now()
    return formatting.format(t)


def calculate_metrics(output, label):
    y_pred = output.data.cpu().numpy()
    y_true = label.data.cpu().numpy()

    pred_label = np.argmax(y_pred, axis=1)
    data = {
        'logloss': torch.nn.CrossEntropyLoss()(output, label).item(),
        'accuracy': accuracy_score(y_true, pred_label)
    }
    return data


if __name__ == '__main__':

    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    train_dataset = CASIADataset(phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model = get_model(opt.backbone)

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.8, nesterov=True)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    now = get_time_string()

    callback = Callbacks([
        TensorboardLogger(log_dir=os.path.join(Config.DATASET_DIR, f'tensorboard_logging/{now}')),
        LoggingCallback()
    ])

    start = time.time()
    for epoch in range(opt.max_epoch):
        scheduler.step()
        model.train()
        callback.on_epoch_start(epoch)

        for i, data in enumerate(trainloader):
            callback.on_batch_start(n_batch=i)
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader) + i

            metric = calculate_metrics(output, label)
            callback.on_batch_end(loss=loss.item(), n_batch=i, train_metric=metric)
            break
        if epoch % opt.save_interval == 0 or epoch == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, epoch)

        model.eval()
        acc, th = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        if opt.display:
            visualizer.display_current_results(iters, acc, name='test_acc')

        callback.on_epoch_end(epoch, valid_metric={'accuracy': acc, 'threshold': th})
