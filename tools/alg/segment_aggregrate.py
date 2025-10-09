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


def aggregate_segment(path, save_dir):
    _, file_name = os.path.split(path)

    image = sitk.ReadImage(path)

    # 进行中值滤波
    # smooth_image = medium_smooth_sitk(image, 3)
    # sitk.WriteImage(smooth_image, os.path.join(save_dir, file_name[:-4] + '_smooth.mhd'))


    smooth_image = sitk.CurvatureFlow(
                        image,
                        timeStep=0.5,
                        numberOfIterations=20
                    )
    # sitk.WriteImage(smooth_image, os.path.join(save_dir, file_name[:-4] + '_smooth_1.mhd'))

    with TimerContext('region_growing_segmentation'):
        image_array = sitk.GetArrayFromImage(image)
        bone_seeds = get_seeds(image_array > 180)
        mask_aggregate = region_growing_segmentation_2(
                smooth_image,
                bone_seeds,
                160,
                255
            )


        # mask_aggregate = sitk.ConfidenceConnected(
        #                 smooth_image,
        #                 seedList=bone_seeds,
        #                 numberOfIterations=1,
        #                 multiplier=0,
        #                 initialNeighborhoodRadius=1,
        #                 replaceValue=1
        #             )

        sitk.WriteImage(mask_aggregate, os.path.join(save_dir, file_name[:-4] + '_mask_aggregate.mhd'))

    # # # 先进行Otsu阈值分割
    # # otsu = sitk.OtsuThreshold(smooth_image)

    # # sitk.WriteImage(otsu, os.path.join(save_dir, file_name[:-4] + '_otsu.mhd'))


    # # 计算梯度幅度
    # gradient = sitk.GradientMagnitude(smooth_image)


    # # gradient = sitk.GradientMagnitudeRecursiveGaussian(
    # #                 smooth_image,
    # #                 sigma=1.0  # 高斯核的标准差
    # #             )

    # sitk.WriteImage(gradient, os.path.join(save_dir, file_name[:-4] + '_gradient_gaussian.mhd'))

    # # 分水岭分割
    # watershed = sitk.MorphologicalWatershed(
    #             gradient,
    #             level=0.7,
    #             markWatershedLine=False,
    #             fullyConnected=True
    #         )


    # sitk.WriteImage(watershed, os.path.join(save_dir, file_name[:-4] + '_watershed.mhd'))


mhd_file = '/home/gzz/dev/data/downsample-4/009_H025_055KaoL_W30_D/009_H025_055KaoL_W30_D_0_fast_1__0.mhd'
save_dir = '/media/gzz/D/aggregrate0608'

os.makedirs(save_dir, exist_ok = True)

aggregate_segment(mhd_file, save_dir)
