from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.transforms import *
from args import argument_parser, image_dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
#from torchreid.losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
#from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers, accuracy, load_pretrained_weights
# from torchreid.utils.reidtools import visualize_ranked_results
# from torchreid.utils.generaltools import set_random_seed
from torchreid.eval_metrics import evaluate
# from torchreid.samplers import RandomIdentitySampler
#from torchreid.optimizers import init_optimizer
# from torchreid.lr_schedulers import init_lr_scheduler
from torchreid.dataset_loader import read_image

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args
    
    #set_random_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    #if args.use_cpu: use_gpu = False
    #log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    #sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        print('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print('Initializing model: {}'.format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'})
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    # Load model here
    model_link = "data/sett2/checkpoint_ep1000.pth.tar"
    if check_isfile(model_link):
        load_pretrained_weights(model, model_link)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    dataset_name = "grozi"
    queryloader = testloader_dict[dataset_name]['query']
    galleryloader = testloader_dict[dataset_name]['gallery']
    test(model, queryloader, galleryloader, use_gpu, return_distmat=True)

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5], return_distmat=False):
    #batch_time = AverageMeter()

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
        print(img_tensor.shape)
        qf, q_pids = [], []
        features = model(img_tensor)
        features = features.data.cpu()
        qf.append(features)
        qf = torch.cat(qf, 0)
        # q_pids.extend(6)
        # q_pids = np.asarray(q_pids)


        # qf, q_pids = [], []
        # for batch_idx, (imgs, pids, camids, path) in enumerate(queryloader):
        #     print(imgs.shape)
        #     if use_gpu:
        #         imgs = imgs.cuda()
            
        #     features = model(imgs)

        #     features = features.data.cpu()
        #     qf.append(features)
        #     q_pids.extend(pids)
        # qf = torch.cat(qf, 0)
        # q_pids = np.asarray(q_pids)
        # print("q_pids")
        # print(q_pids)
        
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
    # print('Computing CMC and mAP')
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    # print('Results ----------')
    # print('mAP: {:.1%}'.format(mAP))
    # print('CMC curve')
    # for r in ranks:
    #     print('Rank-{:<3}: {:.1%}'.format(r, cmc[r-1]))
    # print('------------------')

 
if __name__ == '__main__':
    main()
