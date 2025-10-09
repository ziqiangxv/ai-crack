# -*- coding:utf-8 -*-

''''''

import typing
import SimpleITK as sitk
import numpy as np
from .tensor_utils import Slice, get_axis_by_plane

VOI_TYPE = typing.Sequence[typing.Sequence[int]]

class CrackStatistic(object):
    ''''''

    def __init__(
        self,
        mask_path: str,
        reader: typing.Callable[[str], np.ndarray],
        voi: typing.Optional[typing.Union[VOI_TYPE, str, np.ndarray]] = None,
        labels: typing.Dict[str, int] = {'background': 0, 'crack': 1, 'hole': 2}
        ) -> None:
        ''''''

        self.mask = reader(mask_path)

        assert len(self.mask.shape) == 3

        self.labels = labels

        if voi is not None:
            if type(voi) is VOI_TYPE:
                assert len(voi) == 2 and len(voi[0]) == len(voi[1]) == 3

                z_start, y_start, x_start = voi[0]
                z_end, y_end, x_end = voi[1]

                self.mask = self.mask[z_start:z_end, y_start:y_end, x_start:x_end]

            elif type(voi) is str:
                pass

            elif type(voi) is np.ndarray:
                self.mask[voi == 0] = 0

    def pixel_count_3d(self, label: str) -> float:
        ''''''

        assert label in self.labels.keys(), f'label {label} is not in labels'

        return  np.sum(self.mask == self.labels[label])

    def pixel_count_2d(
        self,
        label: str,
        plane: str,
        index: typing.Optional[int] = None
        ) -> typing.Union[float, typing.Sequence[float]]:
        ''''''

        assert label in self.labels.keys(), f'label {label} is not in labels'

        if index is not None:
            slice = Slice(plane, index)
            return np.sum(slice.get_data(self.mask) == self.labels[label])

        else:
            axis = get_axis_by_plane(plane, layout = 'DHW')
            return np.sum(self.mask == self.labels[label], axis = axis)

    @property
    def SSA(self):
        ''''''
        pass
