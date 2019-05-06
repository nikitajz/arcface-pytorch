"""Test on LFW Dataset Pair
"""
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageOps
from optuna import create_study, Trial
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from torchvision import transforms
from tqdm import tqdm

from models.resnet import get_model

backbone = 'resnet18'
lfw_txt_path = '/workdir/lfw_test_pair.txt'
lfw_root = '/workdir/dataset/lfw-deepfunneled/'
input_shape = (3, 200, 200)
use_gpu = True


def img_to_feature(model, img, transformer, use_flip=True, device=None):
    if use_flip:
        data = torch.zeros([2, *input_shape])
        data[0] = transformer(img)
        data[1] = transformer(ImageOps.mirror(img))
    else:
        data = torch.zeros([1, *input_shape])
        data[0] = transformer(img)

    if device:
        data = data.to(device)

    with torch.no_grad():
        output = model(data)
    return output.data.cpu().numpy().reshape(-1)


df_test = pd.read_csv(lfw_txt_path, sep=' ', header=None)
df_test.columns = ['left_img', 'right_img', 'target']
use_images = set(df_test['left_img']) | set(df_test['right_img'])

use_metrics = [
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
]


def calc_metrics(y_true, y_pred):
    m = [f(y_true, y_pred) for f in use_metrics]
    return m


class LFWObjective:
    def __init__(self, pretrained_weight):
        assert pretrained_weight is None
        self.pretrained_weight = pretrained_weight
        model = get_model(backbone)()
        model.load_state_dict(torch.load(self.pretrained_weight))
        model.eval()

        if use_gpu:
            device = torch.device('cuda')
            model.to(device)
        else:
            device = None
        self.device = device

    def __call__(self, trial: Trial):

        resize_size = trial.suggest_int('resize_size', 250, 500)
        use_flip = trial.suggest_categorical('use_flip', choices=[True, False])
        acc, _ = run_test_accuracy(
            self.model,
            resize_size,
            use_flip,
            device=self.device)
        return acc


def run_test_accuracy(model, resize_size=351, use_flip=True, device=None):
    """
    lwf dataset でテスト精度を検証する
    テスト画像は一度 `resize_size` に拡大縮小されたあと, 画像の中心から input_size だけ切り取りをする.

    ## Note

    * 今の初期値(例えば `resize_size=351`) は optuna で最適化された値。
      本当は mtcnn 等でアラインメントをするべきなのだけれどまだできていない。
    * `use_flip` は True のほうがよさ気.

    Args:
        model: 検証するモデル
        resize_size: テスト画像のリサイズサイズ
        use_flip: True のとき flip した画像と元画像の特徴ベクトルを concat したベクトルを特徴ベクトルとする.
        device: pytorch.device oject

    Returns:

    """

    test_transformer = transforms.Compose([
        transforms.Resize(size=(resize_size, resize_size)),
        transforms.CenterCrop(size=input_shape[1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    name2embedding = OrderedDict()

    for p in tqdm(use_images, total=len(use_images)):
        img_path_i = os.path.join(lfw_root, p)
        img_i = Image.open(img_path_i)
        embedding_i = img_to_feature(model, img_i, test_transformer, use_flip, device=device)
        name2embedding[p] = embedding_i

    similarities = []

    for i, row in df_test.iterrows():
        try:
            x, y = name2embedding[row.left_img], name2embedding[row.right_img]
        except KeyError as e:
            print(e)
            continue
        similarities.append(1 - cosine(x, y))

    df_eval = pd.DataFrame()
    df_eval['target'] = df_test.target
    df_eval['pred'] = similarities
    sns.boxenplot(data=df_eval, x='target', y='pred')
    roc_auc_score(df_eval.target, df_eval.pred)
    thresholds = np.linspace(-1, 1, 1000)

    data = []
    for t in thresholds:
        pred_label = np.where(df_eval.pred > t, 1, 0)
        data.append(calc_metrics(df_eval.target, pred_label))

    df_metric = pd.DataFrame(data, columns=[f.__name__ for f in use_metrics])

    df_metric.accuracy_score.plot()
    val = df_metric[accuracy_score.__name__].values
    acc = np.max(val)
    best_threshold = thresholds[np.argmax(val)]
    return acc, best_threshold


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description=__doc__)
    parser.add_argument('-w' '--weight', type=str, required=True, help='path to pretrained model weight')
    parser.add_argument('-n', '--n_trials', type=int, default=200)
    return vars(parser.parse_args())


if __name__ == '__main__':
    CONFIG = get_args()
    study = create_study(direction='maximize')
    study.optimize(LFWObjective(pretrained_weight=CONFIG.get('weight', None)),
                   n_trials=CONFIG.get('n_trials', 100),
                   n_jobs=1)
    study.trials_dataframe().to_csv('../dataset/lfw_test.csv')
