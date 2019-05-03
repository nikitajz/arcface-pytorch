import os

import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import env


class AbstractDataset(data.Dataset):
    root_path = None
    relative_path = True
    n_classes = None

    def __init__(self,
                 phase='train',
                 input_shape=(1, 128, 128)):

        self.phase = phase
        self.input_shape = input_shape
        self.df_meta = pd.read_csv(os.path.join(self.root_path, 'meta.csv'))
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

    def __str__(self):
        cls_name = self.__class__.__name__
        s = cls_name.replace('Dataset', '')
        s = s.lower()
        return s

    @classmethod
    def name(cls):
        s = cls.__name__
        s = s.replace('Dataset', '')
        s = s.lower()
        return s

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

        if self.relative_path:
            img_path = os.path.join(self.root_path, img_path)
        data = Image.open(img_path)
        data = data.convert(self.img_to)
        data = self.transforms(data)
        return data.float(), label

    def __len__(self):
        return len(self.df_meta)


class CASIADataset(AbstractDataset):
    root_path = os.path.join(env.DATASET_DIR, 'CASIA-WebFace')
    relative_path = False
    n_classes = 912


class CelebaDataset(AbstractDataset):
    root_path = os.path.join(env.DATASET_DIR, 'celeba')
    relative_path = True
    n_classes = 2000


def get_dataset(name, *args, **kwargs) -> AbstractDataset:
    all_datasets = [
        CASIADataset,
        CelebaDataset
    ]

    for d in all_datasets:
        if d.name() == name:
            return d(*args, **kwargs)

    raise ModuleNotFoundError()
