# -*- coding:utf-8 -*-

import env
import os

raw_dir = 'E:\\dev\\src-2\\crack-data\\original'

mhd_dir = os.path.join(os.path.dirname(raw_dir), 'mhd')

os.makedirs(mhd_dir, exist_ok=True)

raws = [f for f in os.listdir(raw_dir) if f.endswith('.raw')]

raw_dir_ = os.path.basename(raw_dir)

for f in raws:
    with open(os.path.join(mhd_dir, f'{f[:-4]}.mhd'), 'w') as file:
        file.writelines(
            'ObjectType = Image\n'
            'NDims = 3\n'
            'DimSize = 2048 2048 1024\n'
            'ElementType = MET_UCHAR\n'
            f'ElementDataFile = ../{raw_dir_}/{f}'
        )
