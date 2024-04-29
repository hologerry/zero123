import json
import math
import os


root_dir = "/data/Dynamics/ScalarFlow/zero123_dataset/"
with open(os.path.join(root_dir, "valid_paths.json")) as f:
    paths = json.load(f)

total_objects = len(paths)
val_paths = paths[: math.floor(total_objects / 100.0 * 1.5)]  # used last 1% as validation
train_paths = paths[math.floor(total_objects / 100.0 * 1.5):]  # used first 99% as training

print(val_paths)


# 99
"""
# 'sim_000102_frame_000154', 'sim_000102_frame_000155', 'sim_000102_frame_000156',
# 'sim_000102_frame_000157', 'sim_000102_frame_000158', 'sim_000102_frame_000159',
# 'sim_000103_frame_000020', 'sim_000103_frame_000021', 'sim_000103_frame_000022',
# 'sim_000103_frame_000023', 'sim_000103_frame_000024', 'sim_000103_frame_000025',
# 'sim_000103_frame_000026', 'sim_000103_frame_000027', 'sim_000103_frame_000028',
# 'sim_000103_frame_000029', 'sim_000103_frame_000030', 'sim_000103_frame_000031',
# 'sim_000103_frame_000032', 'sim_000103_frame_000033', 'sim_000103_frame_000034',
# 'sim_000103_frame_000035', 'sim_000103_frame_000036', 'sim_000103_frame_000037',
# 'sim_000103_frame_000038', 'sim_000103_frame_000039', 'sim_000103_frame_000040',
# 'sim_000103_frame_000041', 'sim_000103_frame_000042', 'sim_000103_frame_000043',
# 'sim_000103_frame_000044', 'sim_000103_frame_000045', 'sim_000103_frame_000046',
# 'sim_000103_frame_000047', 'sim_000103_frame_000048', 'sim_000103_frame_000049',
# 'sim_000103_frame_000050', 'sim_000103_frame_000051', 'sim_000103_frame_000052',
# 'sim_000103_frame_000053', 'sim_000103_frame_000054', 'sim_000103_frame_000055',
# 'sim_000103_frame_000056', 'sim_000103_frame_000057', 'sim_000103_frame_000058',
# 'sim_000103_frame_000059', 'sim_000103_frame_000060', 'sim_000103_frame_000061',
# 'sim_000103_frame_000062', 'sim_000103_frame_000063', 'sim_000103_frame_000064',
# 'sim_000103_frame_000065', 'sim_000103_frame_000066', 'sim_000103_frame_000067',
# 'sim_000103_frame_000068', 'sim_000103_frame_000069', 'sim_000103_frame_000070',
# 'sim_000103_frame_000071', 'sim_000103_frame_000072', 'sim_000103_frame_000073',
# 'sim_000103_frame_000074', 'sim_000103_frame_000075', 'sim_000103_frame_000076',
# 'sim_000103_frame_000077', 'sim_000103_frame_000078', 'sim_000103_frame_000079',
# 'sim_000103_frame_000080', 'sim_000103_frame_000081', 'sim_000103_frame_000082',
# 'sim_000103_frame_000083', 'sim_000103_frame_000084', 'sim_000103_frame_000085',
# 'sim_000103_frame_000086', 'sim_000103_frame_000087', 'sim_000103_frame_000088',
# 'sim_000103_frame_000089', 'sim_000103_frame_000090', 'sim_000103_frame_000091',
# 'sim_000103_frame_000092', 'sim_000103_frame_000093', 'sim_000103_frame_000094',
# 'sim_000103_frame_000095', 'sim_000103_frame_000096', 'sim_000103_frame_000097',
# 'sim_000103_frame_000098', 'sim_000103_frame_000099', 'sim_000103_frame_000100',
# 'sim_000103_frame_000101', 'sim_000103_frame_000102', 'sim_000103_frame_000103',
# 'sim_000103_frame_000104', 'sim_000103_frame_000105', 'sim_000103_frame_000106',
# 'sim_000103_frame_000107', 'sim_000103_frame_000108', 'sim_000103_frame_000109',
# 'sim_000103_frame_000110', 'sim_000103_frame_000111', 'sim_000103_frame_000112',
# 'sim_000103_frame_000113', 'sim_000103_frame_000114', 'sim_000103_frame_000115',
# 'sim_000103_frame_000116', 'sim_000103_frame_000117', 'sim_000103_frame_000118',
# 'sim_000103_frame_000119', 'sim_000103_frame_000120', 'sim_000103_frame_000121',
# 'sim_000103_frame_000122', 'sim_000103_frame_000123', 'sim_000103_frame_000124',
# 'sim_000103_frame_000125', 'sim_000103_frame_000126', 'sim_000103_frame_000127',
# 'sim_000103_frame_000128', 'sim_000103_frame_000129', 'sim_000103_frame_000130',
# 'sim_000103_frame_000131', 'sim_000103_frame_000132', 'sim_000103_frame_000133',
# 'sim_000103_frame_000134', 'sim_000103_frame_000135', 'sim_000103_frame_000136',
# 'sim_000103_frame_000137', 'sim_000103_frame_000138', 'sim_000103_frame_000139',
# 'sim_000103_frame_000140', 'sim_000103_frame_000141', 'sim_000103_frame_000142',
# 'sim_000103_frame_000143', 'sim_000103_frame_000144', 'sim_000103_frame_000145',
# 'sim_000103_frame_000146', 'sim_000103_frame_000147', 'sim_000103_frame_000148',
# 'sim_000103_frame_000149', 'sim_000103_frame_000150', 'sim_000103_frame_000151',
# 'sim_000103_frame_000152', 'sim_000103_frame_000153', 'sim_000103_frame_000154',
# 'sim_000103_frame_000155', 'sim_000103_frame_000156', 'sim_000103_frame_000157',
# 'sim_000103_frame_000158', 'sim_000103_frame_000159'
"""
# 1.5
# 'sim_000000_frame_000020', 'sim_000000_frame_000021', 'sim_000000_frame_000022', 'sim_000000_frame_000023', 'sim_000000_frame_000024', 'sim_000000_frame_000025', 'sim_000000_frame_000026', 'sim_000000_frame_000027', 'sim_000000_frame_000028', 'sim_000000_frame_000029', 'sim_000000_frame_000030', 'sim_000000_frame_000031', 'sim_000000_frame_000032', 'sim_000000_frame_000033', 'sim_000000_frame_000034', 'sim_000000_frame_000035', 'sim_000000_frame_000036', 'sim_000000_frame_000037', 'sim_000000_frame_000038', 'sim_000000_frame_000039', 'sim_000000_frame_000040', 'sim_000000_frame_000041', 'sim_000000_frame_000042', 'sim_000000_frame_000043', 'sim_000000_frame_000044', 'sim_000000_frame_000045', 'sim_000000_frame_000046', 'sim_000000_frame_000047', 'sim_000000_frame_000048', 'sim_000000_frame_000049', 'sim_000000_frame_000050', 'sim_000000_frame_000051', 'sim_000000_frame_000052', 'sim_000000_frame_000053', 'sim_000000_frame_000054', 'sim_000000_frame_000055', 'sim_000000_frame_000056', 'sim_000000_frame_000057', 'sim_000000_frame_000058', 'sim_000000_frame_000059', 'sim_000000_frame_000060', 'sim_000000_frame_000061', 'sim_000000_frame_000062', 'sim_000000_frame_000063', 'sim_000000_frame_000064', 'sim_000000_frame_000065', 'sim_000000_frame_000066', 'sim_000000_frame_000067', 'sim_000000_frame_000068', 'sim_000000_frame_000069', 'sim_000000_frame_000070', 'sim_000000_frame_000071', 'sim_000000_frame_000072', 'sim_000000_frame_000073', 'sim_000000_frame_000074', 'sim_000000_frame_000075', 'sim_000000_frame_000076', 'sim_000000_frame_000077', 'sim_000000_frame_000078', 'sim_000000_frame_000079', 'sim_000000_frame_000080', 'sim_000000_frame_000081', 'sim_000000_frame_000082', 'sim_000000_frame_000083', 'sim_000000_frame_000084', 'sim_000000_frame_000085', 'sim_000000_frame_000086', 'sim_000000_frame_000087', 'sim_000000_frame_000088', 'sim_000000_frame_000089', 'sim_000000_frame_000090', 'sim_000000_frame_000091', 'sim_000000_frame_000092', 'sim_000000_frame_000093', 'sim_000000_frame_000094', 'sim_000000_frame_000095', 'sim_000000_frame_000096', 'sim_000000_frame_000097', 'sim_000000_frame_000098', 'sim_000000_frame_000099', 'sim_000000_frame_000100', 'sim_000000_frame_000101', 'sim_000000_frame_000102', 'sim_000000_frame_000103', 'sim_000000_frame_000104', 'sim_000000_frame_000105', 'sim_000000_frame_000106', 'sim_000000_frame_000107', 'sim_000000_frame_000108', 'sim_000000_frame_000109', 'sim_000000_frame_000110', 'sim_000000_frame_000111', 'sim_000000_frame_000112', 'sim_000000_frame_000113', 'sim_000000_frame_000114', 'sim_000000_frame_000115', 'sim_000000_frame_000116', 'sim_000000_frame_000117', 'sim_000000_frame_000118', 'sim_000000_frame_000119', 'sim_000000_frame_000120', 'sim_000000_frame_000121', 'sim_000000_frame_000122', 'sim_000000_frame_000123', 'sim_000000_frame_000124', 'sim_000000_frame_000125', 'sim_000000_frame_000126', 'sim_000000_frame_000127', 'sim_000000_frame_000128', 'sim_000000_frame_000129', 'sim_000000_frame_000130', 'sim_000000_frame_000131', 'sim_000000_frame_000132', 'sim_000000_frame_000133', 'sim_000000_frame_000134', 'sim_000000_frame_000135', 'sim_000000_frame_000136', 'sim_000000_frame_000137', 'sim_000000_frame_000138', 'sim_000000_frame_000139', 'sim_000000_frame_000140', 'sim_000000_frame_000141', 'sim_000000_frame_000142', 'sim_000000_frame_000143', 'sim_000000_frame_000144', 'sim_000000_frame_000145', 'sim_000000_frame_000146', 'sim_000000_frame_000147', 'sim_000000_frame_000148', 'sim_000000_frame_000149', 'sim_000000_frame_000150', 'sim_000000_frame_000151', 'sim_000000_frame_000152', 'sim_000000_frame_000153', 'sim_000000_frame_000154', 'sim_000000_frame_000155', 'sim_000000_frame_000156', 'sim_000000_frame_000157', 'sim_000000_frame_000158', 'sim_000000_frame_000159', 'sim_000001_frame_000020', 'sim_000001_frame_000021', 'sim_000001_frame_000022', 'sim_000001_frame_000023', 'sim_000001_frame_000024', 'sim_000001_frame_000025', 'sim_000001_frame_000026', 'sim_000001_frame_000027', 'sim_000001_frame_000028', 'sim_000001_frame_000029', 'sim_000001_frame_000030', 'sim_000001_frame_000031', 'sim_000001_frame_000032', 'sim_000001_frame_000033', 'sim_000001_frame_000034', 'sim_000001_frame_000035', 'sim_000001_frame_000036', 'sim_000001_frame_000037', 'sim_000001_frame_000038', 'sim_000001_frame_000039', 'sim_000001_frame_000040', 'sim_000001_frame_000041', 'sim_000001_frame_000042', 'sim_000001_frame_000043', 'sim_000001_frame_000044', 'sim_000001_frame_000045', 'sim_000001_frame_000046', 'sim_000001_frame_000047', 'sim_000001_frame_000048', 'sim_000001_frame_000049', 'sim_000001_frame_000050', 'sim_000001_frame_000051', 'sim_000001_frame_000052', 'sim_000001_frame_000053', 'sim_000001_frame_000054', 'sim_000001_frame_000055', 'sim_000001_frame_000056', 'sim_000001_frame_000057', 'sim_000001_frame_000058', 'sim_000001_frame_000059', 'sim_000001_frame_000060', 'sim_000001_frame_000061', 'sim_000001_frame_000062', 'sim_000001_frame_000063', 'sim_000001_frame_000064', 'sim_000001_frame_000065', 'sim_000001_frame_000066', 'sim_000001_frame_000067', 'sim_000001_frame_000068', 'sim_000001_frame_000069', 'sim_000001_frame_000070', 'sim_000001_frame_000071', 'sim_000001_frame_000072', 'sim_000001_frame_000073', 'sim_000001_frame_000074', 'sim_000001_frame_000075', 'sim_000001_frame_000076', 'sim_000001_frame_000077', 'sim_000001_frame_000078', 'sim_000001_frame_000079', 'sim_000001_frame_000080', 'sim_000001_frame_000081', 'sim_000001_frame_000082', 'sim_000001_frame_000083', 'sim_000001_frame_000084', 'sim_000001_frame_000085', 'sim_000001_frame_000086', 'sim_000001_frame_000087', 'sim_000001_frame_000088', 'sim_000001_frame_000089', 'sim_000001_frame_000090', 'sim_000001_frame_000091', 'sim_000001_frame_000092', 'sim_000001_frame_000093', 'sim_000001_frame_000094', 'sim_000001_frame_000095', 'sim_000001_frame_000096', 'sim_000001_frame_000097'