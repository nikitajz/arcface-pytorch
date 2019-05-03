import os
from glob import glob
from typing import List

import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

import env


class AbstractDataset(data.Dataset):
    root_path = None
    meta_filename = 'meta.csv'
    relative_path = True
    n_classes = None

    label_colname = 'label'
    img_colname = 'img_path'

    def __init__(self,
                 phase='train',
                 input_shape=(1, 128, 128),
                 recreate=False):

        self.phase = phase
        self.input_shape = input_shape
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

        self.df_meta = self.read_metadata(force=recreate)
        self.index_to_data = self.df_meta.to_dict(orient='index')

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
    def meta_path(self):
        return os.path.join(self.root_path, self.meta_filename)

    @property
    def exist_metadata(self):
        return os.path.exists(self.meta_path)

    @property
    def is_greyscale(self):
        return self.input_shape[0] == 1

    @property
    def img_to(self):
        if self.is_greyscale:
            return 'L'
        else:
            return 'RGB'

    def read_metadata(self, force=False):
        if self.exist_metadata and not force:
            return pd.read_csv(self.meta_path)

        print('create metadata')
        df = self.create_metadata()
        df.to_csv(self.meta_path, index=False)
        return df

    def create_metadata(self) -> pd.DataFrame:
        raise NotImplementedError()

    def __getitem__(self, index):
        data = self.index_to_data[index]
        img_path, label = data[self.img_colname], data[self.label_colname]

        if self.relative_path:
            img_path = os.path.join(self.root_path, img_path)
        data = Image.open(img_path)
        data = data.convert(self.img_to)
        data = self.transforms(data)
        return data.float(), label

    def __len__(self):
        return len(self.df_meta)


class CASIAFullDataset(AbstractDataset):
    root_path = os.path.join(env.DATASET_DIR, 'CASIA-WebFace')
    relative_path = False
    meta_filename = 'meta_full.csv'
    min_value_count = 20

    def create_metadata(self):
        img_paths = glob(os.path.join(self.root_path, '*/*.jpg'))
        df_meta = pd.DataFrame(img_paths, columns=['img_path'])
        df_meta['dir_name'] = [str(p.split('/')[-2]) for p in img_paths]
        vc = df_meta.dir_name.value_counts()
        use_dirnames = self.get_use_dirnames(vc)
        df_label = pd.DataFrame(use_dirnames, columns=['dir_name'])
        df_label.index.name = 'label'
        df_label = df_label.reset_index()
        df_meta = pd.merge(df_meta, df_label, on='dir_name', how='right')
        return df_meta

    def get_use_dirnames(self, vc) -> List[str]:
        use = vc[vc >= self.min_value_count]
        return use.index

    @property
    def n_classes(self):
        return len(self.df_meta[self.label_colname].unique())


class CASIADataset(CASIAFullDataset):
    min_value_count = 200


class CelebaDataset(AbstractDataset):
    root_path = os.path.join(env.DATASET_DIR, 'celeba')
    relative_path = True
    n_classes = 2000


def get_dataset(name, *args, **kwargs) -> AbstractDataset:
    all_datasets = [
        CASIAFullDataset,
        CASIADataset,
        CelebaDataset
    ]

    for d in all_datasets:
        if d.name() == name:
            return d(*args, **kwargs)

    raise ModuleNotFoundError()
