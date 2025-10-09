import os
from ai_crack import CrackSegment2D, mhd_reader, mhd_mask_dumper

output_dir = '/media/gzz/D/train-outputs/2d_aggregate/003_7_matrix/mean/'

input_dir = '/media/gzz/D/data/downsample-4/014_H025_130Mont_W30/'

files = ['014_H025_130Mont_W30_D_8_fast_1__0.mhd']

os.makedirs(output_dir, exist_ok=True)

cs = CrackSegment2D(
    model_path = '/media/gzz/D/train_worksapce_2d/checkpoints/003_7_matrix/epoch_200.pth',
    net = 'multi_plane_unet2d_256',
    net_in_channel = 1, #如果训练集里只有单种切面，=1,否则=2，切面种类要么是1要么是3
    net_out_channel = 2,
    plane = 'xyz', # 有三种切面就写xyz
    slice_batch_size = 32,
    save_dir = output_dir,
    input_image_reader = mhd_reader,
    output_mask_dumper = mhd_mask_dumper,
    fusion_tactic = 'mean',
    element_spacing = [1.0, 1.0, 1.0],
    device ='cuda:0'
)


if len(files) == 0:
    for file in os.listdir(input_dir):
        if not file.endswith('.mhd'):
            continue

        print(file)

        cs(os.path.join(input_dir, file))

else:
    for file in files:
        print(file)
        cs(os.path.join(input_dir, file))

