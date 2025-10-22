# -*- coding:utf-8 -*-

''''''

import typing
import torch
from torch.utils.data import Dataset
from ...tensor_utils import normalize_tensor


class ImageDatasetBase(Dataset):
    ''''''

    def __init__(
        self,
        reader: typing.Callable[[str], torch.Tensor],
        transform: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
        device: str = 'cuda:0',
        dtype: str = 'float32',
        ):
        ''''''

        super().__init__()

        self.reader = reader

        self.transform = transform

        self.normalize = normalize

        self.device = device

        self.dtype = dtype


    def get_data(self, image_path, transform_label: str):
        if self.transform is not None and torch.is_grad_enabled():
            data = self.read_data(image_path)

            data = self.transform(data, transform_label)[0]

            if self.normalize and transform_label == 'image':
                data = normalize_tensor(data)

            return data

        else:
            return self.read_data_with_transform(image_path, transform_label)


    def read_data(self, image_path):
        return self.reader(image_path)


    def read_data_with_transform(self, image_path, transform_label):
        data = self.read_data(image_path)

        if self.normalize and transform_label == 'image':
            data = normalize_tensor(data)

        return data
