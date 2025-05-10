import os
from ai_crack import CrackSegment, mhd_reader, mhd_mask_dumper

output_dir = '/home/gzz/dev/vnet128-segment/009_H025_055KaoL_W30_D'

input_dir = '/home/gzz/dev/data/downsample-4/009_H025_055KaoL_W30_D'

cs = CrackSegment(
    model_path = '/home/gzz/dev/train_workspace/checkpoints/001/epoch_30.pth',
    net = 'vnet128',
    infer_tactic = 'slide_window',
    save_dir = output_dir,
    input_image_reader = mhd_reader,
    output_mask_dumper = mhd_mask_dumper,
    element_spacing = [4, 4, 4],
    window_size = [96, 128, 128],
    overlap = [16, 32, 32],
    overlap_fusion_tactic = 'max',
    device ='cuda:0'
)



for file in os.listdir(input_dir):
    if not file.endswith('.mhd'):
        continue

    print(file)

    cs(os.path.join(input_dir, file))

