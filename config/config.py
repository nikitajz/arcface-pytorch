import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def get_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description=__doc__)

    parser.add_argument('--debug', action='store_true',
                        help='If add it, run with debugging mode (not record and stop one batch per epoch')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer Name')
    parser.add_argument('--metric', type=str, default='arc_margin', help='Metrics Name')
    parser.add_argument('--lr', type=float, default=.1, help='learning rate')
    return vars(parser.parse_args())


class Config(object):
    """
    学習/検証を実行する際の設定値
    """
    args = get_arguments()

    is_debug = args.get('debug', False)
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 912
    metric = args.get('metric', None)
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    DATASET_DIR = '/workdir/dataset/'

    # CASIA DATASET のルートディレクトリ
    CASIA_ROOT = os.path.join(DATASET_DIR, 'CASIA-WebFace')

    val_list = '/data/Datasets/webface/val_data_13938.txt'

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = os.path.join(DATASET_DIR, 'lfw-deepfunneled')
    lfw_test_list = os.path.join(lfw_root, 'lfw_test_pair.txt')

    checkpoints_path = os.path.join(DATASET_DIR, 'checkpoints')
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/resnet18_110.pth'

    # 学習に関するパラメータ
    # [TODO] arg で制御したい気分
    save_interval = 10

    train_batch_size = 128  # batch size
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
    weight_decay = 1e-8

    # use in adabound
    final_lr = .1
    amsbound = True
