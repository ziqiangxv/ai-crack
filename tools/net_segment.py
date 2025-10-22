import os
from ai_crack import CrackSegment3D, mhd_reader, mhd_mask_dumper

input_dir = '/home/xzq/dev/zuo/data/train'

output_dir = '/home/xzq/dev/zuo/train_workspace_crack_bubble/001_3d_1009_vnet128_1/segment'

# os.makedirs(output_dir, exist_ok=True)

cs = CrackSegment3D(
    model_path = '/home/xzq/dev/zuo/train_workspace_crack_bubble/001_3d_1009_vnet128_1/checkpoints/epoch_500.pth',
    net = 'multi_plane_vnet_128',
    net_out_channel = 3,
    infer_tactic = 'slide_window',
    save_dir = output_dir,
    input_image_reader = mhd_reader,
    output_mask_dumper = mhd_mask_dumper,
    element_spacing = [4, 4, 4],
    window_size = [96, 128, 128],
    overlap = [16, 32, 32],
    overlap_fusion_tactic = 'average',
    device ='cuda:0'
)

for file in os.listdir(input_dir):
    if not file.endswith('.mhd'):
        continue

    print(file)

    cs(os.path.join(input_dir, file))

