import os
import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import minimum_filter, median_filter
import typing
from utils import TimerContext, timer

def otsu_thresholding(image):
    # 使用 Otsu 方法计算阈值
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    mask = otsu_filter.Execute(image)
    return mask


def manual_thresholding(image, lower_threshold, upper_threshold):
    manual_threshold_filter = sitk.BinaryThresholdImageFilter()
    manual_threshold_filter.SetLowerThreshold(lower_threshold)
    manual_threshold_filter.SetUpperThreshold(upper_threshold)
    manual_threshold_filter.SetInsideValue(1)
    manual_threshold_filter.SetOutsideValue(0)
    mask = manual_threshold_filter.Execute(image)
    return mask

@timer
def region_growing_segmentation(image, seed_points, lower_threshold, upper_threshold):
    region_growing_filter = sitk.ConnectedThresholdImageFilter()
    region_growing_filter.SetSeedList(seed_points)
    region_growing_filter.SetLower(lower_threshold)
    region_growing_filter.SetUpper(upper_threshold)
    mask = region_growing_filter.Execute(image)
    return mask

def get_max_len_slice(vector):
    # 使用双指针法找到最长的连续非零子数组的起始和结束索引
    max_len = 0
    max_start = 0
    max_end = 0
    start = 0
    end = 0

    while end < len(vector):
        if vector[start] == 0:
            start += 1
            end = start
            continue

        if vector[end] == 1:
            if end - start > max_len:
                max_len = end - start
                max_start = start
                max_end = end
            end += 1
        else:
            start = end + 1
            end = start

    return max_start, max_end

def get_first_slice(vector):
    start = 0
    end = 0

    while start < len(vector) and vector[start] == 0:
        start += 1
        end = start

    while end < len(vector) and vector[end] == 1:
        end += 1

    return start, end

@timer
def median_smooth_torch(input_tensor, kernel_size):
    ''''''

    # 添加批量维度和通道维度，以适应 F.median_pool3d 的输入要求
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    # 进行中值滤波
    output = F.median_pool3d(input_tensor, kernel_size=kernel_size)
    # 移除批量维度和通道维度
    output = output.squeeze(0).squeeze(0)

    return output

@timer
def mininum_smooth_torch(input_tensor, kernel_size):
    """
    对三维张量进行最小值滤波
    :param input_tensor: 输入张量，形状为 (height, width, depth)
    :param kernel_size: 滤波窗口大小，整数
    :return: 最小值滤波后的张量
    """
    # 调整输入张量的形状为 (1, 1, height, width, depth)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # 形状变为 (1, 1, height, width, depth)

    # 使用 unfold 将输入张量展开为滑动窗口的形式
    unfolded = F.unfold(input_tensor, kernel_size=(kernel_size, kernel_size, kernel_size), padding=kernel_size // 2)

    unfolded = unfolded.view(1, kernel_size ** 3, -1)

    # 对每个窗口计算最小值
    min_values, _ = unfolded.min(dim=1)  # 沿着窗口维度计算最小值

    # 重塑回原始形状
    min_output = min_values.view_as(input_tensor.squeeze(0).squeeze(0))  # 去掉多余的维度，恢复为 (height, width, depth)

    return min_output

@timer
def medium_smooth_numpy(image: typing.Union[np.ndarray, sitk.Image], filter_size) -> typing.Union[np.ndarray, sitk.Image]:
    if type(image) is np.ndarray:
        return median_filter(image, size=filter_size)

    elif type(image) is sitk.Image:
        array = sitk.GetArrayFromImage(image)
        smooth = median_filter(array, size=filter_size)
        new_image = sitk.GetImageFromArray(smooth)
        new_image.CopyInformation(image)
        return new_image

    else:
        raise Exception('')

@timer
def minimum_smooth_numpy(image: typing.Union[np.ndarray, sitk.Image], filter_size) -> typing.Union[np.ndarray, sitk.Image]:
    if type(image) is np.ndarray:
        return minimum_filter(image, size=filter_size)

    elif type(image) is sitk.Image:
        array = sitk.GetArrayFromImage(image)
        smooth = minimum_filter(array, size=filter_size)
        new_image = sitk.GetImageFromArray(smooth)
        new_image.CopyInformation(image)
        return new_image

    else:
        raise Exception('')

@timer
def medium_smooth_sitk(image: sitk.Image, filter_size: int) -> sitk.Image:
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius([filter_size, filter_size, filter_size])
    smooth_image = median_filter.Execute(image)
    return smooth_image

def get_seeds(binary):
    seeds = np.where(binary)
    return [(int(seeds[2][i]), int(seeds[0][i]), int(seeds[1][i])) for i in range(len(seeds[0]))]


@timer
def crack_segment(path, save_dir):
    image = sitk.ReadImage(path)

    # 进行中值滤波
    smooth_image = medium_smooth_sitk(image, 1)

    with TimerContext('region_growing_segmentation'):
        image_array = sitk.GetArrayFromImage(image)
        bone_seeds = get_seeds(image_array > 200)

        mask_roi = region_growing_segmentation(
                smooth_image,
                bone_seeds,
                140,
                255
            )

    mask_roi_array = sitk.GetArrayFromImage(mask_roi)

    image_array = image_array * mask_roi_array
    image_sum = image_array.sum(axis=(1, 2))

    image_sum_1 = image_sum[: image_sum.shape[0] // 2]
    image_sum_1 -= image_sum_1.min()
    image_sum_1 = image_sum_1 / image_sum_1.max()

    image_sum_2 = image_sum[image_sum.shape[0] // 2:]
    image_sum_2 -= image_sum_2.min()
    image_sum_2 = image_sum_2 / image_sum_2.max()

    _, slice_1 = get_first_slice(image_sum_1 < 0.25)
    _, slice_2 = get_first_slice(np.flip(image_sum_2) < 0.25)
    slice_2 = image_sum.shape[0] - slice_2

    mask_roi_array[:slice_1] = 0
    mask_roi_array[slice_2:] = 0

    mask_roi_ = sitk.GetImageFromArray(mask_roi_array)
    mask_roi_.CopyInformation(mask_roi)
    mask_roi = mask_roi_

    with TimerContext('Hole fill'):
        for _ in range(1):
            fill_hole_filter = sitk.BinaryFillholeImageFilter()
            fill_hole_filter.SetFullyConnected(True)  # False 表示使用四连通，True 表示使用八连通
            mask_roi = fill_hole_filter.Execute(mask_roi)

            structuring_element = sitk.sitkBall
            radius = 20
            structuring_element_size = [radius] * mask_roi.GetDimension()
            mask_roi = sitk.BinaryMorphologicalClosing(mask_roi, structuring_element_size, structuring_element)


    mask_roi_array = sitk.GetArrayFromImage(mask_roi)

    # sitk.WriteImage(mask_roi, os.path.join(save_dir, f[:-4] + '_roi_mask.mhd'))

    # 分割裂纹
    # 最小值滤波
    smooth_image = minimum_smooth_numpy(image, 2)
    image_array = sitk.GetArrayFromImage(smooth_image)
    # sitk.WriteImage(smooth_image, os.path.join(save_dir, f[:-4] + '_smooth.mhd'))

    crack_seeds = get_seeds((image_array > 180) & mask_roi_array)
    mask_crack = region_growing_segmentation(smooth_image, crack_seeds, 110, 255)

    # sitk.WriteImage(mask_crack, os.path.join(save_dir, f[:-4] + '_hard_mask.mhd'))

    mask_crack_array = sitk.GetArrayFromImage(mask_crack)
    mask_crack_array &= mask_roi_array
    mask_crack_array = mask_roi_array - mask_crack_array

    mask_crack_ = sitk.GetImageFromArray(mask_crack_array)
    mask_crack_.CopyInformation(mask_crack)
    mask_crack = mask_crack_

    # with TimerContext('opening'):
    #     structuring_element = sitk.sitkBall
    #     radius = 1
    #     structuring_element_size = [radius] * mask_roi.GetDimension()
    #     mask_crack = sitk.BinaryMorphologicalOpening(mask_crack, structuring_element_size, structuring_element)


    with TimerContext('connection analysis'):
        # 连通域分析
        cc = sitk.ConnectedComponent(mask_crack)
        mask_crack_array = sitk.GetArrayFromImage(mask_crack)
        cc_array = sitk.GetArrayFromImage(cc)

        # 获取所有连通域的标签和体积
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(cc)

        # lables = list(label_stats.GetLabels())
        # volumes = [label_stats.GetPhysicalSize(label) for label in lables]

        bad_lables = []

        for label in label_stats.GetLabels():
            volume = label_stats.GetPhysicalSize(label)
            if volume <= 5000:
                bad_lables.append(label)

        cc_array_flat = cc_array.flatten()
        mask_bad = np.isin(cc_array_flat, bad_lables).reshape(cc_array.shape)
        mask_good = 1 - mask_bad
        mask_crack_array = mask_crack_array * mask_good
        mask_crack_array = mask_crack_array.astype(np.uint8)

        mask_crack_ = sitk.GetImageFromArray(mask_crack_array)
        mask_crack_.CopyInformation(mask_crack)
        mask_crack = mask_crack_

        # sitk.WriteImage(cc, os.path.join(save_dir, f[:-4] + '_crack_cc.mhd'))

    sitk.WriteImage(mask_crack, os.path.join(save_dir, f[:-4] + '.mhd'))



mhd_dir = '/home/xzq/dev/zuo/downsample-4'

save_dir = '/home/xzq/dev/zuo/mask-4'

os.makedirs(save_dir, exist_ok=True)

mhd_files = [f for f in os.listdir(mhd_dir) if f.endswith('.mhd')]

for f in mhd_files:
    # if f != '009_H025_055KaoL_W30_D_1_fast_10__0.mhd':
    #     continue

    print(f)

    crack_segment(os.path.join(mhd_dir, f), save_dir)

