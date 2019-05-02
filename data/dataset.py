import os

import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

from config import Config


class CASIADataset(data.Dataset):

    def __init__(self, phase='train', input_shape=(1, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape
        self.df_meta = pd.read_csv(os.path.join(Config.CASIA_ROOT, 'meta.csv'))
        self.index_to_data = self.df_meta.to_dict(orient='index')

        normalize = T.Normalize(mean=[0.5], std=[0.5])
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    @property
    def is_greyscale(self):
        return self.input_shape[0] == 1

    @property
    def img_to(self):
        if self.is_greyscale:
            return 'L'
        else:
            return 'RGB'

    def __getitem__(self, index):
        data = self.index_to_data[index]
        img_path, label = data['img_path'], data['label']

        data = Image.open(img_path)
        data = data.convert(self.img_to)
        data = self.transforms(data)
        return data.float(), label

    def __len__(self):
        return len(self.df_meta)
