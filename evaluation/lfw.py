import os

import pandas as pd

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

    def __str__(self):
        return 'lfw'
