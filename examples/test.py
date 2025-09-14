# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers_my import ClusterContrastTrainer
from clustercontrast.evaluators_my import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from PIL import Image
import re
start_epoch = best_mAP = 0
from torch.utils.data.sampler import Sampler
import warnings

warnings.filterwarnings("ignore", message="`data_source` argument is not used and will be removed in 2.2.0.")

def get_data(name, data_dir, kind):
    dataset = datasets.create(name, data_dir, kind)
    return dataset

class Preprocessor_my(Dataset):
    def __init__(self, dataset, transform=None):
        super(Preprocessor_my, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.pattern = re.compile(r'/(\d+_c\d+_\d+_\d)_')
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        path, pid, cid, name = self.dataset[index]

        img = []
        for i,p in enumerate(path):
            _img = Image.open(p).convert('RGB')
            if self.transform is not None:
                _img = self.transform(_img)
            img.append(_img)

        img = torch.cat(img, dim=2)
        return img, pid, cid, name

def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = [*dataset.query,*dataset.gallery]

    test_loader = DataLoader(
        Preprocessor_my(testset, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args, num_parts, init_epoch):
    model = models.create(args.arch, pooling_type=args.pooling_type, num_parts=num_parts, init_epoch=init_epoch)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


# generate new dataset and calculate cluster centers
@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    main_worker(args)

def main_worker(args):

    base_dir = f'{args.logs_dir}/{args.kind}'
    start_time = time.monotonic()

    sys.stdout = Logger(osp.join(base_dir, 'test/log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir, args.kind)

    # Create model
    model = create_model(args, len(args.kind), args.init_epoch)

    # Evaluator
    evaluator = Evaluator(model)


    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(base_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, epoch=50, cmc_flag=True)
    del test_loader
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")

    # data
    parser.add_argument('-d', '--dataset', type=str, default='MVTruck', # MVTruck truck
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--num-instances', type=int, default=16,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--kind', type=str, default='FSB', # F S B FS FB SB FSB
                        help="train kind")

    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50_my',
                        choices=models.names())
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--init_epoch', type=int, default=10)
    parser.add_argument('--dual_memory', type=int, default=[500,600])
    parser.add_argument('--resume', type=str, default=None)

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--step', type=int, default=[20,40])

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/xrs/备份')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'test2_3_2'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")

    main()
