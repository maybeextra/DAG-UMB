from __future__ import absolute_import

import torch.nn
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from clustercontrast.models.pooling import build_pooling_layer
from torchvision import models
import cv2
import copy
import os

__weight = {
    18: models.resnet.ResNet18_Weights.DEFAULT,
    34: models.resnet.ResNet34_Weights.DEFAULT,
    50: models.resnet.ResNet50_Weights.DEFAULT,
    101: models.resnet.ResNet101_Weights.DEFAULT,
    152: models.resnet.ResNet152_Weights.DEFAULT,
}

__model = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152,
}


class Pre(nn.Module):
    def __init__(self, resnet):
        super(Pre, self).__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


def load_resnet(depth):
    if depth not in __model:
        raise KeyError("Unsupported depth:", depth)
    resnet = __model[depth](weights=__weight[depth])
    return resnet

def create_batchnorm1d(num_features, requires_grad=True):
    bn = nn.BatchNorm1d(num_features)
    bn.bias.requires_grad_(requires_grad)
    init.constant_(bn.weight, 1)
    init.constant_(bn.bias, 0)
    return bn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力
        self.sa = SpatialAttention(kernel_size)  # 空间注意力

    def forward(self, x):
        x = x * self.ca(x)  # 使用通道注意力加权输入特征图
        x = x * self.sa(x)# 使用空间注意力进一步加权特征图
        return x  # 返回最终的特征图

class ResNet(nn.Module):

    def __init__(self, depth, init_epoch=10, pooling_type='avg', num_parts=None):
        super(ResNet, self).__init__()
        self.num_parts = num_parts
        self.init_epoch = init_epoch

        resnet = load_resnet(depth)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.special = Pre(resnet)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.num_features = resnet.fc.in_features
        del resnet

        self.attention = CBAM(1024)
        self.gap = build_pooling_layer(pooling_type)
        self.feat_bn = create_batchnorm1d(self.num_features, requires_grad=False)

        self.gap_split = nn.ModuleList(
            [build_pooling_layer(pooling_type) for _ in range(self.num_parts)]
        )
        self.feat_bn_split = nn.ModuleList(
            [create_batchnorm1d(self.num_features, requires_grad=False) for _ in range(self.num_parts)]
        )

    def forward(self, x, epoch):
        batch_size = x.size(0)
        x = self.special(x)
        # torch.Size([b, 64, 64, 32 * self.num_parts])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if epoch >= self.init_epoch:
            x = self.attention(x)
        # torch.Size([b, 1024, 16, 8 * self.num_parts])

        x = self.layer4(x)
        # torch.Size([b, 2048, 16, 8 * self.num_parts])

        x_all = self.gap(x).view(batch_size, -1)  # torch.Size([b, 2048, 1, 1]) to torch.Size([b, 2048])
        x_all = self.feat_bn(x_all)  # torch.Size([b, 2048])

        width = x.size(3) // self.num_parts
        x_splits = []
        for i, (gap, bn) in enumerate(zip(self.gap_split, self.feat_bn_split)):
            x_part = x[:,:, : , i * width: ( i + 1 ) * width]
            x_part = gap(x_part).view(batch_size, -1)
            x_part = bn(x_part)
            x_splits.append(x_part.unsqueeze(1))

        if not self.training:
            x_all = F.normalize(x_all)
            x_splits = [F.normalize(x_split) for x_split in x_splits]

        x_splits = torch.cat(x_splits, 1)
        return x_all, x_splits

def resnet18_my(**kwargs):
    return ResNet(18, **kwargs)


def resnet34_my(**kwargs):
    return ResNet(34, **kwargs)


def resnet50_my(**kwargs):
    return ResNet(50, **kwargs)


def resnet101_my(**kwargs):
    return ResNet(101, **kwargs)


def resnet152_my(**kwargs):
    return ResNet(152, **kwargs)

if __name__ == '__main__':
    model = resnet50_my(num_parts=3).cuda()
    model = nn.DataParallel(model)

    print('==> Draw with the best model:')
    from clustercontrast.utils.serialization import load_checkpoint
    checkpoint = load_checkpoint('../../examples/test2_3_2/FSB/model_best.pth.tar')

    model.load_state_dict(checkpoint['state_dict'], strict=False)

    print("==> Load unlabeled dataset")
    from clustercontrast import datasets
    dataset = datasets.create('MVTruck', '/home/xrs/备份', 'FSB')

    from clustercontrast.utils.data import transforms as T
    import re
    def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

        if testset is None:
            testset = [*dataset.query, *dataset.gallery]
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
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
                for i, p in enumerate(path):
                    _img = Image.open(p).convert('RGB')
                    if self.transform is not None:
                        _img = self.transform(_img)
                    img.append(_img)

                img = torch.cat(img, dim=2)
                return img, pid, cid, name
        test_loader = DataLoader(
            Preprocessor_my(testset, transform=test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=True)

        return test_loader

    def extract_cnn_feature(model, inputs, epoch):
        with autocast():
            x_all, x_splits = model(inputs, epoch)
        x_all = x_all.data.cpu()
        x_splits = x_splits.data.cpu()
        return x_all, x_splits


    import collections
    import time
    from clustercontrast.utils.meters import AverageMeter
    from torch.cuda.amp import autocast

    def extract_features(model, data_loader, epoch, print_freq=50):
        model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = collections.OrderedDict()
        features_s = collections.OrderedDict()
        end = time.time()

        with torch.no_grad():
            for i, (images, _, _, names) in enumerate(data_loader):
                data_time.update(time.time() - end)

                x_all, x_splits = extract_cnn_feature(model, images, epoch)
                for name, f_all, f_split in zip(names, x_all, x_splits):
                    features[name] = f_all
                    features_s[name] = f_split

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % print_freq == 0:
                    print('Extract Features: [{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          .format(i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg))

        return features, features_s


    with torch.no_grad():
        print('==> Create pseudo labels for unlabeled data')
        cluster_loader = get_test_loader(dataset, 256, 128,
                                         64, 4, testset=dataset.train)

        features, features_s = extract_features(model, cluster_loader, epoch=50, print_freq=50)
        del cluster_loader

        features = torch.cat([features[name].unsqueeze(0) for img_path, pid, cid, name in dataset.train], 0)
        features_s = torch.cat([features_s[name].unsqueeze(0) for img_path, pid, cid, name in dataset.train], 0)
        from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
        rerank_dist = compute_jaccard_distance(features, k1=30, k2=6)
        from sklearn.cluster import DBSCAN
        cluster = DBSCAN(eps=0.6, min_samples=4, metric='precomputed', n_jobs=-1)
        pseudo_labels = cluster.fit_predict(rerank_dist)

        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        print('==> Statistics for epoch {}: {} clusters'.format(50, num_cluster))
        pseudo_labeled_dataset = []
        for (img_path, pid, cid, name), label in zip(dataset.train, pseudo_labels):
            if label != -1:
                pseudo_labeled_dataset.append((img_path, label, cid, name))
        del rerank_dist
        from clustercontrast.models.cm import ClusterMemory
        ###############################################################################
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
        cluster_features_all = generate_cluster_features(pseudo_labels, features)
        memory_all = ClusterMemory(model.module.num_features, num_cluster, temp=0.05,
                                   momentum=0.8, use_hard=False).cuda()
        memory_all.features = F.normalize(cluster_features_all, dim=1).cuda()
        del features, cluster_features_all

    import matplotlib.pyplot as plt
    import re  # 用于Preprocessor_my的正则匹配
    import numpy as np
    # PyTorch相关
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from examples.cluster_contrast_train_usl_FSB import get_train_loader
    train_loader = get_train_loader(
        dataset, 256, 128,
        4, 4, 4, 100,
        trainset=pseudo_labeled_dataset
    )

    def generate_CAM_from_loader(model, loader, memory_all, save_dir, epoch=100, alpha=0.6):
        """
        直接从DataLoader生成CAM（使用预处理后的图像）
        :param model: 已加载权重的模型
        :param loader: 数据加载器（返回预处理后的图像张量及伪标签）
        :param memory_all: 聚类记忆库实例
        :param save_dir: 结果保存目录
        :param epoch: 需大于模型init_epoch以激活CBAM
        :param alpha: 热力图透明度系数
        """
        model.eval()
        os.makedirs(save_dir, exist_ok=True)

        # 反标准化参数
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        # Hook配置
        features, gradients = [], []

        def forward_hook(module, input, output):
            features.append(output.detach())

        # 注册hook到目标层
        target_layer = model.module.layer4[-1].conv2
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_full_backward_hook(
            lambda module, grad_input, grad_output: gradients.append(grad_output[0].detach())
        )

        # 修改反向传播目标定义部分
        with torch.enable_grad():
            imgs, pseudo_labels, _, names = loader.next()

            imgs = imgs.cuda()
            pseudo_labels = pseudo_labels.cuda()  # 确保标签在GPU上
            batch_size = imgs.size(0)

            # 前向传播
            model.train()
            outputs = model(imgs, epoch=epoch)
            x_all, x_splits = outputs  # 明确获取全局和分块特征

            #########################################################
            # 关键修改：使用聚类中心作为反向传播目标
            #########################################################
            # 获取对应聚类中心特征
            cluster_centers = memory_all.features  # [num_cluster, feature_dim]
            selected_centers = cluster_centers[pseudo_labels]  # [batch_size, feature_dim]

            # 归一化特征并计算相似度
            x_all_normalized = F.normalize(x_all, p=2, dim=1)
            cos_sim = torch.sum(x_all_normalized * selected_centers, dim=1)
            target = cos_sim.sum()

            # 反向传播
            model.zero_grad()
            target.backward()

            # 计算CAM（保持原有逻辑）
            weights = F.adaptive_avg_pool2d(gradients[-1], (1, 1))
            cam = (features[-1] * weights).sum(dim=1)
            cam = F.relu(cam).cpu().numpy()

            # 反标准化图像
            denorm_imgs = (imgs * std + mean).clamp(0, 1)
            denorm_imgs = denorm_imgs.cpu().numpy().transpose(0, 2, 3, 1) * 255

            # 处理每个样本
            for i in range(batch_size):
                # 获取预处理后的图像
                img_np = denorm_imgs[i].astype(np.uint8)
                img_h, img_w = img_np.shape[:2]

                # 调整热力图尺寸
                cam_resized = cv2.resize(cam[i], (img_w, img_h))
                cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)

                # 生成热力图叠加
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)

                # 保存结果（使用loader中的名称）
                save_path = os.path.join(save_dir, f"{names[i]}_cam.jpg")
                cv2.imwrite(save_path, superimposed)

            # 清空缓存
            features.clear()
            gradients.clear()

        # 移除hook
        handle_forward.remove()
        handle_backward.remove()

    generate_CAM_from_loader(model, train_loader, memory_all,'./')