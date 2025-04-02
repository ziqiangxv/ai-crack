# -*- coding:utf-8 -*-

import os
import env

data_dir = ''

for f in os.listdir(data_dir):
    if not f.endswith('.raw'):
        continue

    with open(os.path.join(data_dir, f'{f[:-4]}.mhd'), 'w') as file:
        file.write(
            'ObjectType = Image\n'
            'NDims = 3\n'
            'DimSize = 2048 2048 1024\n'
            'ElementType = MET_UCHAR\n'
            f'ElementDataFile = {f[:-4]}.raw'
        )
