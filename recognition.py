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


class Recognition():

    def __init__(self):
        super(Recognition, self).__init__()

        if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        use_gpu = torch.cuda.is_available()

        if use_gpu:
            #print('Currently using GPU {}'.format(args.gpu_devices))
            cudnn.benchmark = True

        dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
        _, testloader_dict = dm.return_dataloaders()

        self.model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'})
        # Load model here
        model_link = "data/sett2/checkpoint_ep1000.pth.tar"
        if check_isfile(model_link):
            load_pretrained_weights(self.model, model_link)

        self.model = nn.DataParallel(self.model).cuda() if use_gpu else self.model
        dataset_name = "satudora"
        self.galleryloader = testloader_dict[dataset_name]['gallery']
        # Extract feature for training (gallery) set
        self.model.eval()
        self.gf, self.g_pids = self.extract_feature_train_set()

    def extract_feature_test_image(self,test_image):
        with torch.no_grad():
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
            normalize = Normalize(mean=imagenet_mean, std=imagenet_std)
            #test_image = "/Users/phaihoang/Documents/datasets/satudora-product/all/product03_side1_4_0001.jpg"
            img = read_image(test_image)
            transform_test = Compose([
                Resize((256, 256)),
                ToTensor(),
                normalize,
            ])
            img_tensor = transform_test(img)
            img_tensor = img_tensor.unsqueeze(0)
            qf, q_pids = [], []
            features = self.model(img_tensor)
            features = features.data.cpu()
            qf.append(features)
            qf = torch.cat(qf, 0)
            #print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))
            return qf

    def extract_feature_train_set(self):
        with torch.no_grad():
            gf, g_pids = [], []
            for batch_idx, (imgs, pids, camids, _) in enumerate(self.galleryloader):
                features = self.model(imgs)
                features = features.data.cpu()
                gf.append(features)
                g_pids.extend(pids)
            gf = torch.cat(gf, 0)
            g_pids = np.asarray(g_pids)
            #print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))
        return gf, g_pids

    def predict_image(self, test_image, top_k = 5):

        # Extract feature for test (query) image
        self.qf = self.extract_feature_test_image(test_image)
        m, n = self.qf.size(0), self.gf.size(0)
        distmat = torch.pow(self.qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(self.gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, self.qf, self.gf.t())
        distmat = distmat.numpy()
        indices = np.argsort(distmat, axis=1)
        top_k_index = indices[0][:top_k]
        results = []
        for i in top_k_index:
            results.append(self.g_pids[i])
        return results
