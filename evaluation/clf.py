import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cosine
from torchvision import transforms
from tqdm import tqdm

import environments
from .utils import img_to_feature, calc_metrics, use_metrics


class AbstractEvaluation:
    meta_filename = None
    dir_name = None
    label_name = 'label'
    left_colname = 'left'
    right_colname = 'right'

    def __init__(self):
        self.df_meta = pd.read_csv(self.meta_path)

    @property
    def root_path(self):
        return os.path.join(environments.DATASET_DIR, self.dir_name)

    @property
    def meta_path(self):
        return os.path.join(self.root_path, self.meta_filename)

    @property
    def use_images(self):
        return set(self.df_meta[self.left_colname]) | set(self.df_meta[self.right_colname])

    def call(self, model, input_shape, device=None):
        raise NotImplementedError


class CFPEvaluation(AbstractEvaluation):
    dir_name = 'cfp-dataset'
    left_colname = '0'
    right_colname = '1'

    def __init__(self, eval_type='ff'):

        if eval_type not in ['ff', 'fp']:
            raise ValueError(f'Invalid `eval_type`. Allow ff or fp, actually: {eval_type}')

        self.eval_type = eval_type
        self.meta_filename = f'{eval_type}_meta.csv'
        super(CFPEvaluation, self).__init__()

    def __str__(self):
        return f'cfp_{self.eval_type}'

    def get_img_fullpath(self, p):
        return os.path.join(self.root_path, p)

    def call(self, model, input_shape, device=None):
        model.eval()

        if device:
            model.to(device)

        test_transformer = transforms.Compose([
            transforms.CenterCrop(size=input_shape[1:]),
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
            try:
                x, y = name2embedding[row[self.left_colname]], name2embedding[row[self.right_colname]]
            except KeyError as e:
                print(e)
                continue
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
        val = df_metric['accuracy_score'].values
        acc = np.max(val)
        best_threshold = thresholds[np.argmax(val)]
        validate = {}

        validate[f'{self}_acc'] = acc
        validate[f'{self}_threshold'] = best_threshold
        return validate
