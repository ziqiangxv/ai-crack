import os

image_dir = '/home/xzq/dev/data/crak/origin'

gt_dir = '/home/xzq/dev/data/crack/segment-gt'

config_path = '/home/xzq/dev/data/crack/dataset_config.txt'

with open(config_path, 'w') as file:
    file.write(f'IMAGE_DIR:: {image_dir}\nGT_DIR:: {gt_dir}\n')

    for f in os.listdir(image_dir):
        if not f.endswith('.raw'):
            continue

        assert os.path.exists(os.path.join(gt_dir, f)), f'gt file {f} not exists'

        file.write(f'IMAGE:: {f}\nGT:: {f}\n')
