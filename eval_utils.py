#!/usr/bin/python
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import time
import pickle
import argparse
import torch
import torch.nn as nn
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import argparse




def drop_gps_points(gps_data, gps_data_padding, drop_ratio=0.20):
    """
    将GPS数据中指定比例的有效点变成填充值-1，保留剩余的有效点
    
    Args:
        gps_data: GPS数据张量，形状为 [batch_size, seq_len, 2]
        gps_data_padding: 填充掩码，True表示有效点，False表示填充点
        drop_ratio: 丢弃比例，默认0.2（20%的有效点变成填充值）
    
    Returns:
        gps_data: 处理后的GPS数据，部分有效点被替换为-1
    """
    batch_size, seq_len, _ = gps_data.shape
    
    for i in range(batch_size):
        # 找到有效点索引（True表示有效点）
        valid_indices = torch.where(gps_data_padding[i])[0]
        
        if len(valid_indices) > 2:  # 至少有起点、中间点、终点
            # 排除起点和终点
            middle_indices = valid_indices[1:-1]  # 中间点
            num_middle = len(middle_indices)
            num_drop = max(1, int(num_middle * drop_ratio))  # 至少丢弃1个点
            
            if num_drop > 0:
                # 随机选择要丢弃的点
                selected_indices = torch.randperm(num_middle)[:num_drop]
                drop_points = middle_indices[selected_indices]
                
                # 将选中的点替换为填充值-1（两个维度都设置为-1.0）
                gps_data[i, drop_points, 0] = -1.0  # 纬度维度
                gps_data[i, drop_points, 1] = -1.0  # 经度维度
    
    return gps_data

# 简单的目的地预测模型
class DestinationPredictor(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.1):
        super(DestinationPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



def label_norm(y_train):
    """标准化标签数据"""
    mean = torch.mean(y_train)
    std = torch.std(y_train)
    if std < 1e-8:
        std = torch.tensor(1.0, device=y_train.device, dtype=y_train.dtype)
    normalized_y = (y_train - mean) / std
    return normalized_y, mean, std


def pred_unnorm(y_pred, mean, std):
    """反标准化预测数据"""
    # 确保mean和std在CPU上（因为y_pred已经移动到CPU）
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu()
    if isinstance(std, torch.Tensor):
        std = std.cpu()
    return y_pred * std + mean


class MLPReg(nn.Module):
    def __init__(self, input_size, num_layers, activation):
        super(MLPReg, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        self.layers = []
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(input_size, input_size))
        self.layers.append(nn.Linear(input_size, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x).squeeze(1)


def next_batch_index(ds, bs, shuffle=True):
    """生成批次索引"""
    num_batches = math.ceil(ds / bs)
    index = np.arange(ds)
    if shuffle:
        index = np.random.permutation(index)

    for i in range(num_batches):
        if i == num_batches - 1:
            batch_index = index[bs * i:]
        else:
            batch_index = index[bs * i: bs * (i + 1)]
        yield batch_index




def add_gps_noise(gps_data, gps_data_padding, ratio=0.15):
    """添加GPS噪声，排除起点、终点和无效位（偏移量≤100米）"""
    
    batch_size, seq_len, _ = gps_data.shape
    
    # 创建噪声掩码（排除起点、终点和无效位）
    noise_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=gps_data.device)
    
    # 100米偏移，95%置信区间 → 标准差σ=50米
    scale_deg = 50 / 111000  # 纬度缩放因子（固定，1度≈111km）

    for i in range(batch_size):
        # 找到有效点索引（True表示有效点）
        valid_indices = torch.where(gps_data_padding[i])[0]
        
        if len(valid_indices) > 2:  # 至少有起点、中间点、终点
            # 排除起点和终点
            middle_indices = valid_indices[1:-1]  # 中间点
            num_middle = len(middle_indices)
            num_noise = max(1, int(num_middle * ratio))  # 至少1个噪声点
            
            if num_noise > 0:
                # 随机选择噪声点
                selected_indices = torch.randperm(num_middle)[:num_noise]
                noise_points = middle_indices[selected_indices]
                noise_mask[i, noise_points] = True
                
                #只处理当前batch的纬度
                current_lat = gps_data[i, :, 0]  # [seq_len]
                
                #安全处理极点（cos(90°)=0 → 用1e-6避免除0）
                lat_rad = torch.deg2rad(current_lat)
                cos_lat = torch.cos(lat_rad)
                cos_lat = torch.clamp(cos_lat, min=1e-6)  # 安全边界
                
                # 构建当前batch的缩放因子 [seq_len, 2]
                scale = torch.ones((seq_len, 2), device=gps_data.device)
                scale[:, 0] = scale_deg  # 纬度固定缩放
                scale[:, 1] = scale_deg / cos_lat  # 经度动态缩放
                
                # 生成当前batch的噪声 [seq_len, 2]
                noise = torch.randn(seq_len, 2, device=gps_data.device)
                
                #只对当前batch的噪声点应用
                gps_data[i, noise_points] += scale[noise_points] * noise[noise_points]
    
    return gps_data


def add_road_noise(road_seg_data, road_times, road_seg_data_padding=-1, drop_ratio=0.15, replace_ratio=0.15,
                   vocab_min=1, vocab_max=6113, time_min=0, time_max=1439, seed=None):
    """
    对路段序列同时执行两类扰动：
      - 随机丢弃 drop_ratio 比例的中间有效点（不丢弃首尾与填充位），丢弃后用 road_seg_data_padding 填充；
      - 随机替换 replace_ratio 比例的中间有效点（与丢弃位置不重合），替换为随机合法路段 id 和时间。
    返回 (ids, times)
    参数说明见函数签名。
    """
    ids = road_seg_data.clone()
    times = road_times.clone() if road_times is not None else None
    bsz, seqlen = ids.size()

    gen = torch.Generator(device=ids.device)
    if seed is not None:
        gen.manual_seed(int(seed))

    for bi in range(bsz):
        valid_pos = (ids[bi] != road_seg_data_padding).nonzero(as_tuple=True)[0]
        if valid_pos.numel() <= 2:
            continue
        # 候选位置（排除首尾有效位置）
        cand = valid_pos[1:-1]
        m = cand.numel()
        if m == 0:
            continue

        drop_k = int(np.floor(m * drop_ratio))
        replace_k = int(np.floor(m * replace_ratio))
        # 保证至少一个（若希望至少1可启用下一行），否则当 m 小且比例很小可为0
        # drop_k = max(1, drop_k)
        # replace_k = max(1, replace_k)

        # 若超出可选数目，优先保证丢弃，替换数做调整
        if drop_k + replace_k > m:
            replace_k = max(0, m - drop_k)

        total_k = drop_k + replace_k
        if total_k == 0:
            continue

        perm = torch.randperm(m, generator=gen, device=ids.device)[:total_k]
        if drop_k > 0:
            drop_idx = cand[perm[:drop_k]]
            ids[bi, drop_idx] = road_seg_data_padding
            if times is not None:
                times[bi, drop_idx] = road_seg_data_padding
        if replace_k > 0:
            replace_idx = cand[perm[drop_k:drop_k+replace_k]]
            rand_ids = torch.randint(vocab_min, vocab_max + 1, (replace_k,), device=ids.device, generator=gen)
            ids[bi, replace_idx] = rand_ids
            if times is not None:
                rand_times = torch.randint(time_min, time_max + 1, (replace_k,), device=ids.device, generator=gen)
                times[bi, replace_idx] = rand_times

    return ids, times

def calculate_single_similarity(vec1, vec2):
    """计算两个归一化后的向量的余弦相似度"""
    # L2归一化
    #vec1_norm = F.normalize(torch.tensor(vec1), dim=0)
    #vec2_norm = F.normalize(torch.tensor(vec2), dim=0)
    vec1_norm = torch.tensor(vec1)
    vec2_norm = torch.tensor(vec2)
    
    # 计算点积相似度
    similarity = torch.dot(vec1_norm, vec2_norm).item()
    return round(similarity, 4)



