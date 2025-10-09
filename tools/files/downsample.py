import os
import SimpleITK as sitk


def read_and_downsample_mhd(file_path, downsample_factor=2):
    # 读取 mhd 文件
    image = sitk.ReadImage(file_path)

    # 获取原始图像的尺寸
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # 计算降采样后的尺寸
    new_size = [int(size / downsample_factor) for size in original_size]

    # 计算新的像素间距
    new_spacing = [spacing * downsample_factor for spacing in original_spacing]

    # 创建一个仿射变换，这里使用单位矩阵，因为不需要旋转或平移
    transform = sitk.Transform()
    transform.SetIdentity()

    # 使用线性插值进行重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkLinear)

    # 执行重采样
    downsampled_image = resampler.Execute(image)

    return downsampled_image


if __name__ == "__main__":
    mhd_dir = '/home/xzq/dev/zuo/data/mhd'

    save_dir = '/home/xzq/dev/zuo/data/downsample-4'

    os.makedirs(save_dir, exist_ok=True)

    downsample_factor = 4

    mhd_files = [f for f in os.listdir(mhd_dir) if f.endswith('.mhd')]

    for f in mhd_files:
        print(f)

        down_image = read_and_downsample_mhd(os.path.join(mhd_dir, f), downsample_factor)

        sitk.WriteImage(down_image, os.path.join(save_dir, f))
