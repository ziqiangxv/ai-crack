import SimpleITK as sitk
import os

path = '/media/gzz/D/segment/01crack+bubble/GT001-0916/014_H025_130Mont_W30_D_5_fast_1__0.mhd'
save_path = '/media/gzz/D/segment/01crack+bubble/GT001-0916/change/0014_H025_130Mont_W30_D_5_fast_1__0.mhd'

mask = sitk.ReadImage(path)
mask_array = sitk.GetArrayFromImage(mask)
mask_array[mask_array == 3] = 2
mask_new_ = sitk.GetImageFromArray(mask_array)
mask_new_.CopyInformation(mask)
sitk.WriteImage(mask_new_, save_path)