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
from sympy.codegen import Print
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

class RandomMultipleGallerySamplerNoCam(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)

        self.data_source = data_source
        self.index_pid = collections.defaultdict(int)
        self.pid_index = collections.defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cid, _) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            ret.append(i)

            pid_i = self.index_pid[i]
            index = self.pid_index[pid_i]

            select_indexes = No_index(index, i)
            if not select_indexes:
                continue
            if len(select_indexes) >= self.num_instances:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
            else:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

            for kk in ind_indexes:
                ret.append(index[kk])

        return iter(ret)

class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)
        self.data_source = data_source
        self.index_pid = collections.defaultdict(int)
        self.pid_cam = collections.defaultdict(list)
        self.pid_index = collections.defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cid, _) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, _, i_cam, _ = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if not select_indexes:
                    continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return iter(ret)

def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer =  T.Compose([
            T.Resize((height, width), interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            normalizer,
            T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])

    train_set = dataset.train if trainset is None else trainset
    rmgs_flag = num_instances > 0

    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(
            Preprocessor_my(train_set, transform=train_transformer),
            batch_size=batch_size, num_workers=workers, sampler=sampler,
            shuffle=not rmgs_flag, pin_memory=True, drop_last=True
        ), length=iters
    )

    return train_loader


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
    global start_epoch, best_mAP
    start_epoch = 0
    start_time = time.monotonic()

    sys.stdout = Logger(osp.join(base_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir, args.kind)

    # Create model
    model = create_model(args, len(args.kind), args.init_epoch)

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step, gamma=0.1)

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model.load_state_dict(checkpoint['state_dict'])

    # Trainer
    trainer = ClusterContrastTrainer(model)

    for epoch in range(start_epoch,args.epochs):
        if epoch == start_epoch:
            # DBSCAN cluster
            eps = args.eps
            print('Clustering criterion: eps: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=dataset.train)

            features, features_s = extract_features(model, cluster_loader, epoch=epoch, print_freq=50)
            del cluster_loader

            features = torch.cat([features[name].unsqueeze(0) for img_path, pid, cid, name in dataset.train], 0)
            features_s = torch.cat([features_s[name].unsqueeze(0) for img_path, pid, cid, name in dataset.train], 0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
            pseudo_labels = cluster.fit_predict(rerank_dist)

            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))
            pseudo_labeled_dataset = []
            for (img_path, pid, cid, name), label in zip(dataset.train, pseudo_labels):
                if label != -1:
                    pseudo_labeled_dataset.append((img_path, label, cid, name))
            del rerank_dist

            ###############################################################################
            cluster_features_all = generate_cluster_features(pseudo_labels, features)
            memory_all = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                                       momentum=args.momentum, use_hard=args.use_hard).cuda()
            memory_all.features = F.normalize(cluster_features_all, dim=1).cuda()
            del features,cluster_features_all
            ###############################################################################

            ###############################################################################
            memory_split = []
            if args.dual_memory[0] <= epoch < args.dual_memory[1]:
                for i in range(model.module.num_parts):
                    feature_s = features_s[:,i,:]
                    cluster_features_split =  generate_cluster_features(pseudo_labels, feature_s)
                    memory =  ClusterMemory(
                        model.module.num_features, num_cluster,
                        temp=args.temp, momentum=args.momentum, use_hard=args.use_hard
                    ).cuda()
                    memory.features = F.normalize(cluster_features_split, dim=1).cuda()
                    memory_split.append(memory)
            del features_s
            ###############################################################################

        trainer.memory_all = memory_all
        trainer.memory_split = memory_split

        train_loader = get_train_loader(
            dataset, args.height, args.width,
            args.batch_size, args.workers, args.num_instances, iters,
            trainset=pseudo_labeled_dataset, no_cam=args.no_cam
        )

        train_loader.new_epoch()

        trainer.train(
            epoch, train_loader, optimizer, lr_scheduler,
            print_freq=args.print_freq, train_iters=len(train_loader)
        )
        del trainer.memory_all, trainer.memory_split, train_loader

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, epoch=epoch, cmc_flag=False)
            del test_loader
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint(
                {
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,},
                is_best,
                fpath=osp.join(base_dir, 'checkpoint.pth.tar')
            )

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(base_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, epoch=args.epochs, cmc_flag=True)
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
    parser.add_argument('--kind', type=str, default='SB', # F S B FS FB SB FSB
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
                        default=osp.join(working_dir, 'test2_2_1_2'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")

    main()
