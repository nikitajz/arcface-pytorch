import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cosine
from torchvision import transforms
from tqdm import tqdm

from .utils import img_to_feature, calc_metrics, use_metrics


class AbstractBooleanEvaluation:
    """
    二枚の画像を受け取って, それらが同一人物かどうかの判定を行う evaluation class

    ## Description

    * root_path(required):
        メタデータや画像が入っているディレクトリへのパス
        普通メタデータと画像はひとつのディレクトリに入っていると思うので, それを指定します.
        メタデータは roo_path 直下に `meta_filename` として配置してください.

    * meta_filename(required):
        同一人物かどうかのラベルが入っている csv file の名前.
        `label_name`, `left_colname`, `right_colname` の各カラムが入っている必要があります

    * label_name:
        meta_filename で同一人物の行のとき 1, そうでないときに 0 が入っている column の名前.
        初期値は `"label"` です. 違う行を指定する場合は上書きしてください

    * left_colname / right_colname:
        入力される画像へのパスが入ったカラムの名前を指定します. default は `"right"`, `"left"`
        パスは初期設定では root_path からの相対パスが入っている想定で動きます.
        もし絶対パスを指定している場合は `abs_path=True` としてください.

    ## Usage

        class LFWEvaluation(AbstractBooleanEvaluation):
            root_path = os.path.join(environments.DATASET_DIR, 'lfw-deepfunneled')
            left_colname = '0'
            right_colname = '1'
            label_name = '2'

            def read_metadata(self):
                return pd.read_csv(self.meta_path, sep=' ', header=None)
    """

    meta_filename = None
    root_path = None
    label_name = 'label'
    left_colname = 'left'
    right_colname = 'right'
    abs_path = False

    # NOTE: 今は metric を変えても名前が変わらない. metric の名前に合わせた名前に変更するようなロジックに変えること
    evaluation_metric = 'accuracy_score'

    def __init__(self):
        self.df_meta = self.read_metadata()
        self.validate_settings()

    def read_metadata(self):
        return pd.read_csv(self.meta_path)

    @property
    def meta_path(self):
        return os.path.join(self.root_path, self.meta_filename)

    @property
    def use_images(self):
        return set(self.df_meta[self.left_colname]) | set(self.df_meta[self.right_colname])

    def get_img_fullpath(self, img_path):
        if self.abs_path:
            return img_path
        return os.path.join(self.root_path, img_path)

    def validate_settings(self):
        # 指定されたカラムが存在しているかどうかのチェック
        for c in [self.left_colname, self.right_colname, self.label_name]:
            if c not in self.df_meta.columns:
                raise ValueError(f'Column {c} is not found on metadata.')

        if len(self.use_images) == 0:
            raise ValueError(f'use images count is Zero. Check your metadata {self.meta_path} columns')

        # 画像ファイルがちゃんと有るかどうかのチェック
        for img in self.use_images:
            img_path = self.get_img_fullpath(img)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f'Image file {img_path} is not found')

            if not os.path.isfile(img_path):
                raise ValueError(f'{img_path} is not file. (maybe dir)')

    def call(self, model, input_shape, device=None):
        model.eval()

        if device:
            model.to(device)

        test_transformer = transforms.Compose([
            # transforms.CenterCrop(size=input_shape[1:]),
            transforms.Resize(size=input_shape[1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        name2embedding = OrderedDict()

        for p in tqdm(self.use_images, total=len(self.use_images)):
            img_path_i = self.get_img_fullpath(p)
            img_i = Image.open(img_path_i)
            embedding_i = img_to_feature(model, img_i, test_transformer, input_shape=input_shape,
                                         use_flip=True, device=device)
            name2embedding[p] = embedding_i

        similarities = []

        for i, row in self.df_meta.iterrows():
            x, y = name2embedding[row[self.left_colname]], name2embedding[row[self.right_colname]]
            similarities.append(1 - cosine(x, y))

        df_eval = pd.DataFrame()
        df_eval['target'] = self.df_meta[self.label_name]
        df_eval['pred'] = similarities
        thresholds = np.linspace(-1, 1, 1000)

        data = []
        for t in thresholds:
            pred_label = np.where(df_eval.pred > t, 1, 0)
            data.append(calc_metrics(df_eval.target, pred_label))

        df_metric = pd.DataFrame(data, columns=[f.__name__ for f in use_metrics])
        val = df_metric[self.evaluation_metric].values
        acc = np.max(val)
        best_threshold = thresholds[np.argmax(val)]
        validate = {}

        validate[f'{self}_acc'] = acc
        validate[f'{self}_threshold'] = best_threshold
        return validate
