import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os
import imageio
#from scipy import misc
from datetime import datetime
from model import UNet
from dataset import test_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("models/model.pt")
model.eval()
model.to(device)

data_path = '/content/drive/MyDrive/data/'
# valset = ['ECSSD', 'HKUIS', 'PASCAL', 'DUT-OMRON', 'THUR15K', 'DUTS-TEST']
valset = ['TrainSet_StaticAndVideo']
for dataset in valset:
    save_path = './saliency_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = data_path + dataset + '/RGB/'
    gt_root = data_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, testsize=352)

    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.array(gt).astype('float')
            gt = gt / (gt.max() + 1e-8)
            image = Variable(image).cuda()
            
            res= model(image)

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            imageio.imwrite(save_path + name + '.png', res)
