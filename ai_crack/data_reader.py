# -*- coding:utf-8 -*-

''''''

import typing
import torch
import SimpleITK as sitk

def mhd_reader(mhd_path: str) -> torch.Tensor:
    image = sitk.ReadImage(mhd_path)
    image = sitk.GetArrayFromImage(image)
    image = torch.from_numpy(image)
    return image.unsqueeze(0)

def mhd_mask_dumper(
    image: torch.Tensor, 
    save_mhd_path: str, 
    element_spacing: typing.Tuple[float, float, float]) -> None:
    image = image.squeeze()
    image = image.numpy().astype('uint8')
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(element_spacing)

    sitk.WriteImage(image, save_mhd_path)
