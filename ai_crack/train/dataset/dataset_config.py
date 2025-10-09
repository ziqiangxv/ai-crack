# -*- coding:utf-8 -*-

''''''

import typing
import os
from ...tensor_utils import Slice

class DatasetConfig(object):
    ''''''

    def __init__(self, config_path: str):
        ''''''

        self.config_path = config_path

        self.image_dir = None

        self.gt_dir = None

        self.images: typing.Sequence[str] = []

        self.gts: typing.Sequence[str] = []

        self.slices: typing.Sequence[typing.Sequence[Slice]] = []

        self.parse()

    def parse(self):
        ''''''

        kw_image_dir    = 'IMAGE_DIR::'
        kw_gt_dir       = 'GT_DIR::'
        kw_image        = 'IMAGE::'
        kw_gt           = 'GT::'
        kw_slice        = 'SLICE::'

        with open(self.config_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                line = line.strip()

                if line == '' or line.startswith('#'):
                    continue

                if line.startswith(kw_image_dir):
                    self.image_dir = line[len(kw_image_dir) :].strip()

                    assert os.path.exists(self.image_dir), f'{kw_image_dir} {self.image_dir} not exists'

                elif line.startswith(kw_gt_dir):
                    self.gt_dir = line[len(kw_gt_dir) :].strip()

                    assert os.path.exists(self.gt_dir), f'{kw_gt_dir} {self.gt_dir} not exists'

                elif line.startswith(kw_image):
                    assert self.image_dir is not None, 'IMAGE_DIR is not set'

                    assert len(self.images) == len(self.gts), 'GT is not set'

                    self.images.append(os.path.join(self.image_dir, line[len(kw_image) :].strip()))

                elif line.startswith(kw_gt):
                    assert self.gt_dir is not None, 'GT_DIR is not set'

                    assert len(self.gts) == len(self.images) - 1, 'IMAGE is not set while setting GT'

                    self.gts.append(os.path.join(self.gt_dir, line[len(kw_gt) :].strip()))

                elif line.startswith(kw_slice):
                    slice_str = line[len(kw_slice) :].strip()
                    slice_parts = slice_str.split('|')

                    if len(self.slices) == len(self.images) - 1:
                        slices = []

                        self.slices.append(slices)

                    elif len(self.slices) == len(self.images):
                        slices = self.slices[-1]

                    else:
                        raise Exception(f'Wrong slice: {line}, slice num mismatch image num')

                    for s in slice_parts:
                        parts = [p.strip() for p in s.split(',')]

                        for _i in range(1, len(parts)):
                            if ':' in parts[_i]:
                                assert parts[_i].count(':') == 2, f"wrong line: {parts[_i]}"

                                if '[' in parts[_i]:
                                    assert ']' in parts[_i], f"wrong line: {parts[_i]}"

                                    roi_str = parts[_i][parts[_i].index('[') + 1 : parts[_i].index(']')]

                                    roi_parts = roi_str.split('&')

                                    assert len(roi_parts) == 4, f"wrong line: {parts[_i]}"

                                    roi = []

                                    for r in roi_parts:
                                        assert r.isdigit(), f"wrong line: {parts[_i]}"

                                        roi.append(int(r.strip()) - 1)

                                    roi = self._reorder_roi(parts[0], roi)

                                    seq_str = parts[_i][: parts[_i].index('[')]

                                else:
                                    seq_str = parts[_i]

                                    roi = None

                                range_parts = seq_str.split(':')

                                assert len(range_parts) == 3, f"wrong line: {seq_str}"

                                for r in range_parts:
                                    assert r.isdigit(), f"wrong line: {seq_str}"

                                range_parts = [int(p.strip()) for p in range_parts]

                                assert range_parts[0] > 0 and range_parts[1] >= range_parts[0] and \
                                    range_parts[2] > 0 and range_parts[1] - range_parts[0] >= range_parts[2], \
                                    f"wrong line: {seq_str}"

                                for i in range(range_parts[0] - 1, range_parts[1], range_parts[2]):
                                    slices.append(Slice(parts[0], i, roi))

                            else:
                                if '[' in parts[_i]:
                                    assert ']' in parts[_i], f"wrong line: {parts[_i]}"

                                    roi_str = parts[_i][parts[_i].index('[') + 1 : parts[_i].index(']')]

                                    roi_parts = roi_str.split('&')

                                    assert len(roi_parts) == 4, f"wrong line: {parts[_i]}"

                                    roi = []

                                    for r in roi_parts:
                                        assert r.isdigit(), f"wrong line: {parts[_i]}"

                                        roi.append(int(r.strip()) - 1)

                                    roi = self._reorder_roi(parts[0], roi)

                                    index_str = parts[_i][: parts[_i].index('[')]

                                else:
                                    assert parts[_i].isdigit(), f"wrong line: {parts[_i]}"

                                    index_str = parts[_i]

                                    roi = None

                                slices.append(Slice(parts[0], int(index_str) - 1, roi))

                else:
                    raise Exception(f'Wrong keyword: {line}')

        assert len(self.images) == len(self.gts), 'IMAGE and GT count not match'

    def _reorder_roi(self, plane: str, roi: typing.List[int]) -> typing.List[int]:
        assert len(roi) == 4, 'ROI must be 4'

        assert roi[3] > roi[1] and roi[2] > roi[0], 'left top and right bottom meaning wrong'

        return [roi[1], roi[0], roi[3], roi[2]]
