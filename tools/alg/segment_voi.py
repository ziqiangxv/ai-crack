import os
import SimpleITK as sitk
import numpy as np
from utils import TimerContext, timer

@timer
def medium_smooth_sitk(image: sitk.Image, filter_size: int) -> sitk.Image:
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius([filter_size, filter_size, filter_size])
    smooth_image = median_filter.Execute(image)
    return smooth_image

@timer
def region_growing_segmentation(image, seed_points, lower_threshold, upper_threshold):
    region_growing_filter = sitk.ConnectedThresholdImageFilter()
    region_growing_filter.SetSeedList(seed_points)
    region_growing_filter.SetLower(lower_threshold)
    region_growing_filter.SetUpper(upper_threshold)
    mask = region_growing_filter.Execute(image)
    return mask

@timer
def region_growing_segmentation_1(image, seed_points):
    # 创建区域生长滤波器
    conf_filter = sitk.ConfidenceConnectedImageFilter()

    # 设置参数
    conf_filter.SetSeedList(seed_points)  # 设置种子点
    conf_filter.SetNumberOfIterations(1)   # 迭代次数
    conf_filter.SetMultiplier(0)         # 设置动态范围因子c，控制灰度范围宽窄
    conf_filter.SetInitialNeighborhoodRadius(1)  # 初始邻域半径
    conf_filter.SetReplaceValue(1)         # 替换值
    mask = conf_filter.Execute(image)
    return mask

@timer
def region_growing_segmentation_2(image, seed_points, lower_threshold, upper_threshold):
    filter = sitk.NeighborhoodConnectedImageFilter()
    filter.SetLower(lower_threshold)
    filter.SetUpper(upper_threshold)
    filter.SetSeedList(seed_points)
    filter.SetReplaceValue(1)
    mask = filter.Execute(image)
    return mask


def get_first_slice(vector):
    start = 0
    end = 0

    while start < len(vector) and vector[start] == 0:
        start += 1
        end = start

    while end < len(vector) and vector[end] == 1:
        end += 1

    return start, end

def get_seeds(binary, seed_num = None):
    seeds = np.where(binary)

    if seed_num is None:
        seed_num = len(seeds[0])

    if seed_num > len(seeds[0]):
        seed_num = len(seeds[0])

    return [(int(seeds[2][i]), int(seeds[1][i]), int(seeds[0][i])) for i in range(seed_num)]


def voi_segment(path, save_dir, thresh_highlight_percentage_1, thresh_highlight_percentage_2, lock_z_begin, lock_z_end):
    image = sitk.ReadImage(path)

    # 进行中值滤波
    smooth_image = medium_smooth_sitk(image, 3)
    # smooth_image = image

    with TimerContext('region_growing_segmentation'):
        image_array = sitk.GetArrayFromImage(image)
        bone_seeds = get_seeds(image_array > 150)
        mask_voi = region_growing_segmentation_2(
                smooth_image,
                bone_seeds,
                128,
                255
            )

    mask_voi_array = sitk.GetArrayFromImage(mask_voi)

    image_array = image_array * mask_voi_array
    image_sum = image_array.sum(axis=(1, 2))

    image_sum_1 = image_sum[: image_sum.shape[0] // 2]
    image_sum_1 -= image_sum_1.min()
    image_sum_1 = image_sum_1 / image_sum_1.max()

    image_sum_2 = image_sum[image_sum.shape[0] // 2:]
    image_sum_2 -= image_sum_2.min()
    image_sum_2 = image_sum_2 / image_sum_2.max()

    _, slice_1 = get_first_slice(image_sum_1 <= thresh_highlight_percentage_1)
    _, slice_2 = get_first_slice(np.flip(image_sum_2) <= thresh_highlight_percentage_2)
    slice_2 = image_sum.shape[0] - slice_2

    mask_voi_array[:slice_1] = 0
    mask_voi_array[slice_2:] = 0

    mask_voi_ = sitk.GetImageFromArray(mask_voi_array)
    mask_voi_.CopyInformation(mask_voi)
    mask_voi = mask_voi_

    with TimerContext('Closing'):
        for _ in range(5):
            fill_hole_filter = sitk.BinaryFillholeImageFilter()
            fill_hole_filter.SetFullyConnected(True)  # False 表示使用四连通，True 表示使用八连通
            mask_voi = fill_hole_filter.Execute(mask_voi)

            structuring_element = sitk.sitkBall
            radius = 3
            structuring_element_size = [radius] * mask_voi.GetDimension()
            mask_voi = sitk.BinaryMorphologicalClosing(mask_voi, structuring_element_size, structuring_element)

    inverted_mask_roi = sitk.Not(mask_voi)


    with TimerContext('Internal Hole Fill'):
        inverted_mask_roi = sitk.Not(mask_voi)
        # 连通域分析
        cc_ = sitk.ConnectedComponent(inverted_mask_roi)
        cc_array_ = sitk.GetArrayFromImage(cc_)

        # 获取所有连通域的标签和体积
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(cc_)

        labels = list(label_stats.GetLabels())
        volumes = [label_stats.GetPhysicalSize(label) for label in labels]
        max_volume_label = labels[np.argmax(volumes)]
        interal_hole_mask = (cc_array_ != max_volume_label).astype(np.uint8)

        mask_voi_array = sitk.GetArrayFromImage(mask_voi)
        mask_voi_array = np.logical_or(mask_voi_array, interal_hole_mask).astype(np.uint8)

        mask_voi_ = sitk.GetImageFromArray(mask_voi_array)
        mask_voi_.CopyInformation(mask_voi)
        mask_voi = mask_voi_

    with TimerContext('Outer Hole Fill'):
        shape = smooth_image.GetSize()
        x, y, z = shape

        mask_voi_array = sitk.GetArrayFromImage(mask_voi)
        sum_array = np.sum(mask_voi_array, axis=(1, 2))
        max_slice_index = np.argmax(sum_array)

        fusion_start = max_slice_index - 50
        fusion_end = max_slice_index + 20

        fusion_array = np.zeros_like(mask_voi_array)

        for i in range(fusion_start, fusion_end):
            _slice_array = mask_voi_array[i, ...]
            _slice_array = np.expand_dims(_slice_array, axis=0)
            _slice = sitk.GetImageFromArray(_slice_array)
            _cc= sitk.ConnectedComponent(_slice)
            _cc_array = sitk.GetArrayFromImage(_cc)

            label_stats = sitk.LabelShapeStatisticsImageFilter()
            label_stats.Execute(_cc)

            labels = list(label_stats.GetLabels())

            if len(labels) > 0:
                volumes = [label_stats.GetPhysicalSize(label) for label in labels]
                max_volume_label = labels[np.argmax(volumes)]

                for label in labels:
                    if label == max_volume_label:
                        continue

                    _cc_array[_cc_array == label] = 0

                fusion_array[i, ...][_cc_array[0, ...] > 0] = 1

        # fusion = sitk.GetImageFromArray(fusion_array)
        # sitk.WriteImage(fusion, os.path.join(save_dir, 'fusion.mhd'))

        fusion_slice = np.max(fusion_array, axis=0)

        mask_voi_array[slice_1 : lock_z_begin] = fusion_slice
        mask_voi_array[0 : lock_z_end] = 0
          
          
        corner_seeds = []

        for i in (0, x):
            for j in (0, y):
                for k in (0, z):
                    corner_seeds.append((i, j, k))

        mask_corner = region_growing_segmentation_2(
                smooth_image,
                corner_seeds,
                110,
                112
            )
        # sitk.WriteImage(mask_corner, os.path.join(save_dir, 'mask_outer.mhd'))


        corner_distance_map = sitk.SignedMaurerDistanceMap(mask_corner, insideIsPositive=False)
        corner_distance_map = sitk.Divide(corner_distance_map, float(sitk.GetArrayFromImage(corner_distance_map).max()))
        # sitk.WriteImage(corner_distance_map, os.path.join(save_dir, 'corner_distance_map.mhd'))

        corner_distance_map_array = sitk.GetArrayFromImage(corner_distance_map)
        mask_container_array = corner_distance_map_array < 1e-3 * 1.4
        mask_voi_array[mask_container_array ==1] = 0


        mask_voi_ = sitk.GetImageFromArray(mask_voi_array)
        mask_voi_.CopyInformation(mask_voi)
        mask_voi = mask_voi_

    _, file_name = os.path.split(path)

    sitk.WriteImage(mask_voi, os.path.join(save_dir, file_name[:-4] + '_voi.mhd'))


mhd_file = '/home/gzz/dev/data/downsample-4/014_H025_130Mont_W30/014_H025_130Mont_W30_D_1_fast_1__0.mhd'
save_dir = '/media/gzz/D/segment-voi'

thresh_highlight_percentage_1 = 0
thresh_highlight_percentage_2 = 0.3
lock_z_begin = 211
lock_z_end = 15

os.makedirs(save_dir, exist_ok = True)

voi_segment(mhd_file, save_dir, thresh_highlight_percentage_2, thresh_highlight_percentage_1, lock_z_begin, lock_z_end)
