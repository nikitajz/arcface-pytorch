import os

import pandas as pd
from torchvision import transforms

import environments
from .base import AbstractBooleanEvaluation


class LFWEvaluation(AbstractBooleanEvaluation):
    root_path = os.path.join(environments.DATASET_DIR, 'lfw-deepfunneled')
    meta_filename = 'lfw_test_pair.txt'
    left_colname = 0
    right_colname = 1
    label_name = 2

    def read_metadata(self):
        return pd.read_csv(self.meta_path, sep=' ', header=None)

    def get_test_transformer(self, input_shape):
        # hack: optuna で良かった resize and center clop の大きさ
        # 本当は MTCNN で顔検出して切り出ししたい
        resize_size = 351
        return transforms.Compose([
            transforms.Resize(size=(resize_size, resize_size)),
            transforms.CenterCrop(size=input_shape[1:]),
            transforms.ToTensor(),
        ])

    def __str__(self):
        return 'lfw'
