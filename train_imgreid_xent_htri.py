from __future__ import print_function
from __future__ import division

import os

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.transforms import *
from args import argument_parser, image_dataset_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.utils.iotools import check_isfile
from torchreid.utils.torchtools import count_num_param,load_pretrained_weights
from torchreid.dataset_loader import read_image

# global variables
parser = argument_parser()
args = parser.parse_args()

        
def main():

    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        print('Currently using CPU, however, GPU is highly recommended')

    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    _, testloader_dict = dm.return_dataloaders()

    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'})
    # Load model here
    model_link = "data/sett2/checkpoint_ep1000.pth.tar"
    if check_isfile(model_link):
        load_pretrained_weights(model, model_link)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    dataset_name = "grozi"
    galleryloader = testloader_dict[dataset_name]['gallery']
    test(model, galleryloader, use_gpu)

def test(model, galleryloader, use_gpu):

    model.eval()

    with torch.no_grad():
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        normalize = Normalize(mean=imagenet_mean, std=imagenet_std)
        test_image = "/Users/phaihoang/Documents/datasets/satudora-product/all/product03_side1_4_0001.jpg"
        img = read_image(test_image)
        transform_test = Compose([
            Resize((96, 96)),
            ToTensor(),
            normalize,
        ])  
        img_tensor = transform_test(img) 
        img_tensor = img_tensor.unsqueeze(0)     
        qf, q_pids = [], []
        features = model(img_tensor)
        features = features.data.cpu()
        qf.append(features)
        qf = torch.cat(qf, 0)
        
        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids = [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            features = model(imgs)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))


    #computing_start = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    top_5_index = indices[0][:5]
    print(top_5_index)
    top_5 = []
    for i in top_5_index:
        top_5.append(g_pids[i])
    print(top_5)

if __name__ == '__main__':
    main()
