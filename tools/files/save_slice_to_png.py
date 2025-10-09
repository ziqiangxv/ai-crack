import SimpleITK as sitk
import numpy as np
import os
from PIL import Image

def convert_mhd_to_png(mhd_path, output_dir):
    """
    将 MHD 文件的所有横切面保存为 PNG 图像

    参数:
    mhd_path: MHD 文件路径
    output_dir: PNG 输出目录
    """

    try:
        # 读取 MHD 文件
        image = sitk.ReadImage(mhd_path)

        # 获取图像数组 (Z, Y, X 顺序)
        image_array = sitk.GetArrayFromImage(image)

        print(f"图像尺寸: {image_array.shape} (切片数, 高度, 宽度)")
        print(f"像素值范围: {np.min(image_array)} - {np.max(image_array)}")

        # 处理每个横切面
        for slice_idx in range(image_array.shape[0]):
            # 获取当前切片
            slice_data = image_array[slice_idx, :, :]

            # 创建 PIL 图像并保存
            img = Image.fromarray(slice_data)
            output_path = os.path.join(output_dir, f"slice_{slice_idx:03d}.png")
            img.save(output_path)

        print(f"成功保存 {image_array.shape[0]} 个切片到 {output_dir}")

    except Exception as e:
        print(f"处理出错: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 输入你的 MHD 文件路径和输出目录
    mhd_file = "/home/xzq/dev/zuo/downsample-4/009_H025_055KaoL_W30_D/009_H025_055KaoL_W30_D_0_fast_1__0.mhd"  # 替换为你的 MHD 文件路径
    output_directory = "/home/xzq/dev/zuo/png/009_H025_055KaoL_W30_D/009_H025_055KaoL_W30_D_0_fast_1__0"   # 替换为你的输出目录

    os.makedirs(output_directory, exist_ok=True)

    convert_mhd_to_png(mhd_file, output_directory)