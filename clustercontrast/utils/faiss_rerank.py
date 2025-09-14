#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
import gc
import faiss

import torch
import torch.nn.functional as F

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
                            index_init_gpu, index_init_cpu
from concurrent.futures import ThreadPoolExecutor

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


# def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=1, use_float16=False):
#     end = time.time()
#     if print_flag:
#         print('Computing jaccard distance...')
#
#     ngpus = faiss.get_num_gpus()
#     N = target_features.size(0)
#     mat_type = np.float16 if use_float16 else np.float32
#
#     if (search_option==0):
#         # GPU + PyTorch CUDA Tensors (1)
#         res = faiss.StandardGpuResources()
#         res.setDefaultNullStreamAllDevices()
#         _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
#         initial_rank = initial_rank.cpu().numpy()
#     elif (search_option==1):
#         # GPU + PyTorch CUDA Tensors (2)
#         res = faiss.StandardGpuResources()
#         index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
#         index.add(target_features.cpu().numpy())
#         _, initial_rank = search_index_pytorch(index, target_features, k1)
#         res.syncDefaultStreamCurrentDevice()
#         initial_rank = initial_rank.cpu().numpy()
#     elif (search_option==2):
#         # GPU
#         index = index_init_gpu(ngpus, target_features.size(-1))
#         index.add(target_features.cpu().numpy())
#         _, initial_rank = index.search(target_features.cpu().numpy(), k1)
#     else:
#         # CPU
#         index = index_init_cpu(target_features.size(-1))
#         index.add(target_features.cpu().numpy())
#         _, initial_rank = index.search(target_features.cpu().numpy(), k1)
#
#
#     nn_k1 = []
#     nn_k1_half = []
#     for i in range(N):
#         nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
#         nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))
#
#     V = np.zeros((N, N), dtype=mat_type)
#     for i in range(N):
#         k_reciprocal_index = nn_k1[i]
#         k_reciprocal_expansion_index = k_reciprocal_index
#         for candidate in k_reciprocal_index:
#             candidate_k_reciprocal_index = nn_k1_half[candidate]
#             if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
#                 k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
#
#         k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
#         dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
#         if use_float16:
#             V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
#         else:
#             V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()
#
#     del nn_k1, nn_k1_half
#
#     if k2 != 1:
#         V_qe = np.zeros_like(V, dtype=mat_type)
#         for i in range(N):
#             V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
#         V = V_qe
#         del V_qe
#
#     del initial_rank
#
#     invIndex = []
#     for i in range(N):
#         invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num
#
#     jaccard_dist = np.zeros((N, N), dtype=mat_type)
#     for i in range(N):
#         temp_min = np.zeros((1, N), dtype=mat_type)
#         # temp_max = np.zeros((1,N), dtype=mat_type)
#         indNonZero = np.where(V[i, :] != 0)[0]
#         indImages = []
#         indImages = [invIndex[ind] for ind in indNonZero]
#         for j in range(len(indNonZero)):
#             temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
#             # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
#
#         jaccard_dist[i] = 1-temp_min/(2-temp_min)
#         # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)
#
#     del invIndex, V
#
#     pos_bool = (jaccard_dist < 0)
#     jaccard_dist[pos_bool] = 0.0
#     if print_flag:
#         print("Jaccard distance computing time cost: {}".format(time.time()-end))
#
#     return jaccard_dist


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
    index.add(target_features.numpy())
    _, initial_rank = search_index_pytorch(index, target_features, k1)
    res.syncDefaultStreamCurrentDevice()
    initial_rank = initial_rank.numpy()

    nn_k1 = []
    nn_k1_half = []
    # 创建了两个空列表 nn_k1 和 nn_k1_half，用于存储每个图像的 k 近邻和 k/2 近邻。
    # 通过一个循环迭代数据集中的每个图像，从 0 到 N-1。

    for i in range(N):
        # 在每次迭代中，使用 k_reciprocal_neigh 函数来获取当前图像的 k 互相近邻和 k/2 互相近邻，并将它们分别添加到 nn_k1 和 nn_k1_half 列表中。
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, round(k1 / 2)))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique

        # 余弦距离转马氏距离
        dist = 2 - 2 * torch.mm(target_features[i].unsqueeze(0).contiguous(),
                                target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    # 如果 k2 不等于 1，则进行查询扩展。
    if k2 != 1:
        # 创建一个与 V 矩阵具有相同形状和数据类型的零矩阵 V_qe。
        V_qe = np.zeros_like(V, dtype=mat_type)
        # 循环遍历数据集中的每个图像，将其索引存储在变量 i 中。
        for i in range(N):
            # 对于每个图像 i，从 initial_rank 中获取其前 k2 个近邻的索引，然后从 V 矩阵中获取这些近邻的行，并计算它们的平均值。这个平均值将作为查询扩展后的相似性得分。
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)

        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]

        for j in range(len(indNonZero)):
            _temp_min = np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + _temp_min

        jaccard_dist[i] = 1 - (temp_min / (2 - temp_min))
    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    del pos_bool

    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time() - end))

    return jaccard_dist
