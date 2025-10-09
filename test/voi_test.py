import numpy as np
import SimpleITK as sitk
import os


slice = np.zeros((512, 512))

slices = []

for i in range(256):
    slices.append(slice + i)

data = np.stack(slices, axis=0).astype(np.uint8)

image = sitk.GetImageFromArray(data)

this_dir = os.path.abspath(os.path.dirname(__file__))

sitk.WriteImage(image, os.path.join(this_dir, 'image.mhd'))
