import os
from ai_crack import CrackSegment2D, mhd_reader, mhd_mask_dumper

output_dir = '/home/xzq/dev/zuo/train_workspace_crack_bubble/unet2d_128_1/segment'

input_dir = '/home/xzq/dev/zuo/data/train'

files = []

os.makedirs(output_dir, exist_ok=True)

cs = CrackSegment2D(
    model_path = '/home/xzq/dev/zuo/train_workspace_crack_bubble/unet2d_128_1/checkpoints/epoch_1908.pth',
    net = 'multi_plane_unet2d_128',
    net_in_channel = 1, #如果训练集里只有单种切面，=1,否则=2，切面种类要么是1要么是3
    net_out_channel = 3,
    plane = 'xyz', # 有三种切面就写xyz
    slice_batch_size = 16,
    save_dir = output_dir,
    input_image_reader = mhd_reader,
    output_mask_dumper = mhd_mask_dumper,
    fusion_tactic = 'mean',
    element_spacing = [1.0, 1.0, 1.0],
    device ='cuda:0',
    dtype = 'float16',
    use_amp = True,
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

