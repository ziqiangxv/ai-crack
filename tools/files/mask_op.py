from ai_crack.data_reader import mhd_reader, mhd_mask_dumper

voi_mask_path = '/media/gzz/D/segment/04_voi/voi_GT/009_H025_055KaoL_W30_D_0_fast_1__0_voi.mhd'
crack_mask_path = '/media/gzz/D/train-outputs/vnet128-segment/Vnet004/sample009/009_H025_055KaoL_W30_D_0_fast_1__0.mhd'
aggerate_mask_path = '/media/gzz/D/train-outputs/2d_aggregate/001_10/009_H025_055KaoL_W30_D_0_fast_1__0.mhd'

save_plaque_mask_path = '/media/gzz/D/segment/02_matrix/009_H025_055KaoL_W30_D_0_fast_1__0.mhd'

voi_mask = mhd_reader(voi_mask_path)
crack_mask = mhd_reader(crack_mask_path)
aggerate_mask = mhd_reader(aggerate_mask_path)

assert voi_mask.shape == crack_mask.shape == aggerate_mask.shape, 'voi_mask, crack_mask, aggerate_mask shape not match'

plaque_mask = voi_mask
plaque_mask[crack_mask == 1] = 0
plaque_mask[aggerate_mask == 1] = 0

mhd_mask_dumper(plaque_mask, save_plaque_mask_path, (1., 1., 1.))

