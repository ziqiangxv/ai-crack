import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ai_crack import CrackStatistic, mhd_reader_numpy



voi_mask_file = '/media/gzz/D/segment/04_voi/voi_GT/009_H025_055KaoL_W30_D_0_fast_1__0_voi.mhd'

target_image_dir = '/home/gzz/dev/data/downsample-4/009_H025_055KaoL_W30_D'

target_mask_dir = '/media/gzz/D/train-outputs/2d_aggregate/002_3/sample009'

save_dir = '/media/gzz/D/stastic_data_outputs/aggerates'

target_label = 'aggerate'

labels = {'background': 0, 'aggerate': 1}

KW1 = '_W30_D_'

KW2 = '_fast_'

target_files = []

os.makedirs(save_dir, exist_ok = True)

if not target_files:
    target_files = [name for name in os.listdir(target_mask_dir) if name.endswith('.mhd')]

voi_mask = mhd_reader_numpy(voi_mask_file)


def get_index(name):
    # 009_H025_055KaoL_W30_D_0_fast_1__0.mhd
    i = name.find(KW1)
    j = name.find(KW2)

    i_start = i + len(KW1)
    i_end = i_start

    j_start = j + len(KW2)
    j_end = j_start

    for _i in range(i + len(KW1), len(name)):
        if name[_i] == '_':
            i_end = _i
            break

    for _j in range(j + len(KW2), len(name)):
        if name[_j] == '_':
            j_end = _j
            break

    return int(name[i_start : i_end]) + 0.1 * int(name[j_start : j_end])

def time_evolution(label):
    print('****************** time evolution')

    time_dump_dir = os.path.join(save_dir, 'time_evolution')
    os.makedirs(time_dump_dir, exist_ok = True)

    voi_volume = np.sum(voi_mask == 1)

    x = []
    y = []
    res = {}

    for file in target_files:
        print(file)

        crack_statistic = CrackStatistic(
            os.path.join(target_mask_dir, file),
            mhd_reader_numpy,
            voi = voi_mask,
            labels = labels
        )

        index = get_index(file)
        percentage = crack_statistic.pixel_count_3d(label) / voi_volume

        x.append(index)
        y.append(percentage)

        res[file] = percentage

    columns = ['File', 'Stage', 'Percentage']
    data = [(_t, _x, _y) for _t, _x, _y in zip(target_files, x, y)]

    df = pd.DataFrame(data = data, columns = columns)
    df.to_csv(os.path.join(time_dump_dir, 'time_evolution.csv'), index = False)

    plt.scatter(x, y)
    plt.xlabel('Stage')
    plt.ylabel('Percentage')
    plt.grid(True)
    plt.savefig(os.path.join(time_dump_dir, 'time_evolution.png'), dpi = 300)

def layer_evolution(label, plane):
    print('****************** layer evolution')

    voi_layer_pixels = np.sum(voi_mask, axis = (1, 2))

    voi_layer_mask = voi_layer_pixels > 0

    layer_dump_dir = os.path.join(save_dir, 'layer_evolution')

    layer_csv_dump_dir = os.path.join(layer_dump_dir, 'csv')

    layer_pixel_ratio_dump_dir = os.path.join(layer_dump_dir, 'pixel_ratio')

    layer_mean_value_dump_dir = os.path.join(layer_dump_dir, 'mean_value')

    os.makedirs(layer_dump_dir, exist_ok = True)

    os.makedirs(layer_csv_dump_dir, exist_ok = True)

    os.makedirs(layer_pixel_ratio_dump_dir, exist_ok = True)

    os.makedirs(layer_mean_value_dump_dir, exist_ok = True)

    columns = ['Layer', 'Layer Percentage', 'Layer Mean Value']

    for i, file in enumerate(target_files):
        print(file)

        image = mhd_reader_numpy(os.path.join(target_image_dir, file))
        mask = mhd_reader_numpy(os.path.join(target_mask_dir, file))

        file_name, file_suffix = os.path.splitext(file)

        crack_statistic = CrackStatistic(
            os.path.join(target_mask_dir, file),
            mhd_reader_numpy,
            voi = voi_mask,
            labels = labels
        )

        layer_percentage = crack_statistic.pixel_count_2d(label, plane)[voi_layer_mask] / voi_layer_pixels[voi_layer_mask] 

        mean_slice_values = np.sum(image * voi_mask * mask, axis = (1, 2))[voi_layer_mask] / np.sum(voi_mask * mask, axis = (1, 2))[voi_layer_mask]

        data = [(i, _p, mean_slice_values[i]) for i, _p in enumerate(layer_percentage)]

        df = pd.DataFrame(data = data, columns = columns)
        df.to_csv(os.path.join(layer_csv_dump_dir, f'{file_name}.csv'), index = False)

        plt.scatter(list(range(len(layer_percentage))), layer_percentage)
        plt.xlabel('Layer')
        plt.ylabel('Percentage')
        plt.grid(True)
        plt.savefig(os.path.join(layer_pixel_ratio_dump_dir, f'{file_name}.png'), dpi = 300)
        plt.clf()
        plt.close()

        plt.scatter(list(range(len(mean_slice_values))), mean_slice_values)
        plt.xlabel('Layer')
        plt.ylabel('Mean Value')
        plt.grid(True)
        plt.savefig(os.path.join(layer_mean_value_dump_dir, f'{file_name}.png'), dpi = 300)
        plt.clf()
        plt.close()

time_evolution(target_label)
layer_evolution(target_label, 'xy')
