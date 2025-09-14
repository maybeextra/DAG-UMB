from __future__ import print_function, absolute_import

import collections
import time
import torch

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from torch.cuda.amp import autocast

def extract_cnn_feature(model, inputs, epoch):
    with autocast():
        x_all, x_splits = model(inputs, epoch)
    x_all = x_all.data.cpu()
    x_splits = x_splits.data.cpu()
    return x_all, x_splits

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


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[name].unsqueeze(0) for img_path, pid, cid, name in query], 0)
    y = torch.cat([features[name].unsqueeze(0) for img_path, pid, cid, name in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for img_path, pid, cid, name in query]
        gallery_ids = [pid for img_path, pid, cid, name in gallery]
        query_cams = [cid for img_path, pid, cid, name in query]
        gallery_cams = [cid for img_path, pid, cid, name in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


import os
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision import transforms

def visualize_topk(distmat, query, gallery, save_dir, topk=10, num_queries=4):
    os.makedirs(save_dir, exist_ok=True)
    query_indices = np.random.choice(len(query), num_queries, replace=False)

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for q_idx in query_indices:
        # 处理查询图像
        q_info = query[q_idx]
        q_path, q_id = q_info[0], q_info[1]

        q_imgs = []
        for p in q_path:
            img = Image.open(p).convert('RGB')
            q_imgs.append(transform(img))
        q_img = torch.cat(q_imgs, dim=2)

        # 转换为OpenCV格式 (不添加标题)
        q_img = q_img.cpu().numpy()
        q_img = np.transpose(q_img, (1, 2, 0))
        q_img = (q_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        q_img = np.clip(q_img * 255, 0, 255).astype(np.uint8)
        q_img = cv2.cvtColor(q_img, cv2.COLOR_RGB2BGR)
        h_query, w_query = q_img.shape[:2]  # 记录原始高度 (256)

        # 获取前topk结果
        g_indices = np.argsort(distmat[q_idx])[:topk]

        # 处理图库图像
        gallery_imgs = []
        for rank, g_idx in enumerate(g_indices, 1):
            g_info = gallery[g_idx]
            g_path, g_id = g_info[0], g_info[1]

            g_imgs = []
            for p in g_path:
                img = Image.open(p).convert('RGB')
                g_imgs.append(transform(img))
            g_img = torch.cat(g_imgs, dim=2)

            # 转换为OpenCV格式
            g_img = g_img.cpu().numpy()
            g_img = np.transpose(g_img, (1, 2, 0))
            g_img = (g_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            g_img = np.clip(g_img * 255, 0, 255).astype(np.uint8)
            g_img = cv2.cvtColor(g_img, cv2.COLOR_RGB2BGR)

            # 添加边框（统一增加高度到262）
            border_color = (0, 255, 0) if q_id == g_id else (0, 0, 255)
            g_img = cv2.copyMakeBorder(g_img,
                                       top=3, bottom=3, left=3, right=3,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=border_color)

            # 调整图库图像高度与查询图像一致（262 → 256）
            g_img = cv2.resize(g_img, (g_img.shape[1], h_query))  # 关键修复

            # 添加排名标注
            cv2.putText(g_img, f"#{rank}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            gallery_imgs.append(g_img)

        # 构建可视化布局
        h, w = h_query, w_query  # 使用统一高度256
        spacer = np.ones((h, 10, 3), dtype=np.uint8) * 255

        # 拼接所有图像
        row = [q_img, spacer]
        for img in gallery_imgs:
            row.extend([img, spacer.copy()])
        vis_row = np.concatenate(row[:-1], axis=1)

        # 添加标题（在最终图像上添加，不影响高度）
        vis_row = cv2.copyMakeBorder(vis_row, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(vis_row, f"Query ID: {q_id}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # 保存结果
        save_path = os.path.join(save_dir, f'query{q_idx}_top{topk}.jpg')
        cv2.imwrite(save_path, vis_row)

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, epoch, cmc_flag=False, rerank=False):
        features, features_s = extract_features(self.model, data_loader, epoch)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if cmc_flag:
            visualize_topk(
                distmat=distmat.numpy(),  # 距离矩阵
                query=query,  # 查询集元数据
                gallery=gallery,  # 图库元数据
                save_dir='./vis_results',  # 保存路径
                topk=6,  # 显示前10个结果
                num_queries=6  # 随机选择4个查询
            )

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
