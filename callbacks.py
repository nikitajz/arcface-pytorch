import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

from utils.logger import get_logger


class AbstractCallback:
    def on_epoch_start(self, epoch: int):
        pass

    def on_epoch_end(self, epoch: int, valid_metric: dict):
        pass

    def on_batch_start(self, n_batch, *args, **kwargs):
        pass

    def on_batch_end(self, loss: float, n_batch, output: np.ndarray, label: np.ndarray):
        pass


class TensorboardLogger(AbstractCallback):
    def __init__(self, log_dir=None):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.current_epoch = -1
        self.history = []

    def on_epoch_start(self, epoch):
        self.current_epoch = epoch
        self.epoch_score_df = pd.DataFrame()

    def calculate_metrics(self, loss, y_true, y_pred):
        # Memo: この計算ロジックの実装は tensorboard 以外でも使うのでここに書くのはおかしい.
        acc = accuracy_score(y_true, y_pred)
        df = pd.DataFrame([loss, acc], index=['loss', 'accuracy']).T
        return df

    def on_batch_end(self, loss, n_batch, output, label):
        pred_label = np.argmax(output, axis=1)
        df_i = self.calculate_metrics(loss, label, pred_label)
        self.epoch_score_df = pd.concat([self.epoch_score_df, df_i], ignore_index=True)

    def on_epoch_end(self, epoch, valid_metric: dict):
        train_metric = self.epoch_score_df.mean().to_dict()
        self.writer.add_scalars('train', tag_scalar_dict=train_metric, global_step=epoch)
        self.writer.add_scalars('validation', tag_scalar_dict=valid_metric, global_step=epoch)


class LoggingCallback(AbstractCallback):
    def __init__(self, log_freq=50):
        self.log_freq = log_freq
        self.logger = get_logger(name=__name__)
        self.epoch = 0
        self.losses = []

    def on_epoch_start(self, epoch: int):
        self.epoch = epoch

    def on_batch_end(self, loss: float, n_batch, output: np.ndarray, label: np.ndarray):
        self.losses.append(loss)

        if n_batch % self.log_freq != 0:
            return

        loss = np.mean(self.losses)
        self.losses = []
        self.logger.info(f'[epoch:{self.epoch:04d}] batch:{n_batch:05d} loss: {loss:.3f}')

    def on_epoch_end(self, epoch: int, valid_metric: dict):
        s = []
        for k, v in valid_metric.items():
            s.append(f'{k} {v:.3f}')

        s = ' '.join(s)
        self.logger.info(f'[validate] {s}')


class Callbacks:
    def __init__(self, callbacks: List[AbstractCallback]):
        self.callbacks = callbacks

    def on_epoch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_epoch_start(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_epoch_end(*args, **kwargs)

    def on_batch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_batch_start(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_batch_end(*args, **kwargs)
