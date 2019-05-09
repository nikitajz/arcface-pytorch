import os

import environments
from .base import AbstractBooleanEvaluation


class CFPEvaluation(AbstractBooleanEvaluation):
    root_path = os.path.join(environments.DATASET_DIR, 'cfp-dataset')
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
