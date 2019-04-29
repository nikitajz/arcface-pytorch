import os


class Config(object):
    """
    学習/検証を実行する際の設定値
    """

    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 13938
    metric = 'arc_margin'
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

    input_shape = (1, 128, 128)

    # optimizer name: `"adam"`, `"sgd"`, `"adabound"`
    optimizer = 'adabound'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = .01  # initial learning rate
    final_lr = .1
    amsbound = True
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-8
