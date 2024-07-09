import torch 

model = torch.load('/ai/volume/Ultra-Fast-Lane-Detection-V2/work_dirs/pth/culane_res18.pth', map_location='cpu')
# model2 = torch.load('/ai/volume/mmsegmentation/work_dirs/culane_xt/iter_111120.pth', map_location='cpu')
model2 = torch.load('/ai/volume/mmsegmentation/work_dirs/culane_xt/iter_27780.pth', map_location='cpu')
model3 = {}

for key1, key2 in zip(model['model'].keys(), model2['state_dict'].keys()):
    print(model['model'][key1].shape, model2['state_dict'][key2].shape)
    model3[key2] = model['model'][key1]

# torch.save(model3, '/ai/volume/mmsegmentation/work_dirs/culane_xt/convert.pth')

import pdb; pdb.set_trace()