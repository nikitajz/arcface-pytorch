import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime


def get_time_string(t=None, formatting='{0:%Y-%m-%d_%H-%M-%S}'):
    if t is None:
        t = datetime.now()
    return formatting.format(t)


def get_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description=__doc__)

    parser.add_argument('--debug', action='store_true',
                        help='If add it, run with debugging mode (not record and stop one batch per epoch')
    # model setting
    parser.add_argument('--dataset', type=str, default='casia', help='dataset name')
    parser.add_argument('--metric', type=str, default='arc_margin', help='Metrics Name')
    parser.add_argument('--weight', type=str, default=None, help='Pretrained model weight path')

    # optimizer settings
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer Name')
    parser.add_argument('--lr', type=float, default=.1, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=.1,
                        help='final learning rate (only activa on `optimizer="adabound"`')

    # run settings
    parser.add_argument('--batch', type=int, default=128, help='training batch size')
    return vars(parser.parse_args())


class Config(object):
    """
    学習/検証を実行する際の設定値
    """
    args = get_arguments()
    now = get_time_string()

    is_debug = args.get('debug', False)
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    dataset = args.get('dataset', None)
    metric = args.get('metric', None)
    easy_margin = False
    use_se = False
    loss = 'logloss'

    pretrained_model_path = args.get('weight', None)

    display = False
    finetune = False

    DATASET_DIR = '/workdir/dataset/'

    # CASIA DATASET のルートディレクトリ
    CASIA_ROOT = os.path.join(DATASET_DIR, 'CASIA-WebFace')
    lfw_root = os.path.join(DATASET_DIR, 'lfw-deepfunneled')
    lfw_test_list = os.path.join(lfw_root, 'lfw_test_pair.txt')

    # 学習に関するパラメータ
    # [TODO] arg で制御したい気分
    save_interval = 10

    train_batch_size = args.get('batch', 32)  # batch size
    test_batch_size = 64

    input_shape = (3, 200, 200)

    # optimizer name: `"adam"`, `"sgd"`, `"adabound"`
    optimizer = args.get('optimizer', None)

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 8  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = args.get('lr', 0.1)  # initial learning rate
    lr_step = 20  # cut lr frequency
    weight_decay = 5e-4

    # use in adabound
    final_lr = args.get('final_lr', .2)
    amsbound = True

    checkpoints_path = os.path.join(DATASET_DIR, 'checkpoints', f'{dataset}_{optimizer}_{now}')
    config_path = os.path.join(checkpoints_path, 'train_config.json')
