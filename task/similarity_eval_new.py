import os
import random
import faiss
import numpy as np
import pandas as pd
import json
import torch
from typing import Dict, Tuple
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 设置线程数
torch.set_num_threads(5)

# 设置CUDA设备
dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
if torch.cuda.is_available():
    torch.cuda.set_device(dev_id)


def get_gps_mean_std(lat_lists, lng_lists):
    all_lats = []
    all_lngs = []
    for lat, lng in zip(lat_lists, lng_lists):
        all_lats.extend(lat)
        all_lngs.extend(lng)
    mean_lat = np.mean(all_lats,axis=0)
    mean_lng = np.mean(all_lngs,axis=0)
    std_lat = np.std(all_lats,axis=0)
    std_lng = np.std(all_lngs,axis=0)

    # 避免标准差为0
    std_lat = np.maximum(std_lat, 1e-6)
    std_lng = np.maximum(std_lng, 1e-6)

    return mean_lat, mean_lng, std_lat, std_lng

def normalize_gps(gps_data, mean_lat, mean_lng, std_lat, std_lng):
    """
    归一化GPS数据
    """
    normalized_gps = gps_data.clone()
    print(normalized_gps[0, 0, 0])
    print(normalized_gps[0,0, 1])
    normalized_gps[:, :, 0] = (gps_data[:, :, 0] - mean_lat) / std_lat
    normalized_gps[:, :, 1] = (gps_data[:, :, 1] - mean_lng) / std_lng
    # 最终限制范围
    normalized_gps = torch.clamp(normalized_gps,min=-3.0, max=3.0)

    return normalized_gps

# FAISS索引构建
def build_faiss_index(features: np.ndarray) -> faiss.IndexFlatIP:  
    """使用JGRM的FAISS索引构建方法"""  



    index = faiss.IndexFlatIP(features.shape[1])  
    index.add(features)  
    return index  

def search_and_evaluate(query_features: np.ndarray,   
                        D_features: np.ndarray,  
                        y: np.ndarray) -> Dict[str, float]:  
    # 构建索引
    index = build_faiss_index(D_features)
    # 检查重复样本示例
    #print(f"查询0在数据库中的出现次数: {np.sum(np.all(query_features == query_features[0], axis=1))}")

    # 执行搜索 - 搜索top-1000结果
    D, I = index.search(query_features, min(1000, len(D_features)))
    print(D[:2, :10])

    print("\n前十个查询查到的轨迹序号:")
    for i in range(min(10, len(I))):
        print(f"查询 {i+1}: {I[i][:10]}")  # 显示每个查询的前10个结果

    # 计算评估指标
    hit1 = 0
    hit5 = 0
    hit3 = 0
    hit10 = 0
    rank_sum = 0
    no_hit = 0
    for i, r in enumerate(I):
        # 检查真实标签是否在搜索结果中
        print(f"Evaluating query {i+1}/{len(query_features)}", end='\r')
        if y[i] in r:

            # 计算排名（索引位置）
            rank = np.where(r == y[i])[0][0]
            rank_sum += (rank+1)
            # 根据排名更新命中计数
            if rank < 1:
                hit1 += 1
            if rank < 3:
                hit3 += 1
            if rank < 5:
                hit5 += 1
            if rank < 10:
                hit10 += 1
        else:
            no_hit += 1
    num_queries = len(query_features)

    mean_rank = rank_sum / (num_queries - no_hit) if (num_queries - no_hit) > 0 else 0
    hr1 = hit1 / (num_queries - no_hit) if (num_queries - no_hit) > 0 else 0
    hr3 = hit3 / (num_queries - no_hit) if (num_queries - no_hit) > 0 else 0
    hr5 = hit5 / (num_queries - no_hit) if (num_queries - no_hit) > 0 else 0
    hr_10 = hit10 / (num_queries - no_hit) if (num_queries - no_hit) > 0 else 0
      
    return {
        'Mean Rank': mean_rank,
        'HR@1': hr1,
        'HR@3': hr3,
        'HR@5': hr5,
        'HR@10': hr_10,
        'No Hit': no_hit
    }

def perturb_route_data(route_list):
    """
    对路段数据进行扰动，随机丢弃10%到20%的点（不包含起点和终点）
    """
    if len(route_list) <= 2:  # 如果路段长度小于等于2，无法进行扰动
        return route_list
    random.seed(2023)  # 使用系统时间作为随机种子
    # 计算要丢弃的点数（10%到20%之间，不包含起点和终点）
    num_points_to_drop = random.randint(
        max(1, int(len(route_list) * 0.10)),  # 至少丢弃1个点
        max(1, int(len(route_list) * 0.10))   # 最多丢弃20%的点
    )
    
    # 确保不丢弃起点和终点
    available_indices = list(range(1, len(route_list) - 1))  # 可丢弃的点的索引（不包含首尾）
    
    if len(available_indices) == 0:  # 如果没有可丢弃的点
        return route_list
    
    # 随机选择要丢弃的点
    indices_to_drop = random.sample(available_indices, min(num_points_to_drop, len(available_indices)))
    
    # 构建扰动后的路段
    perturbed_route = []
    for i, point in enumerate(route_list):
        if i not in indices_to_drop:  # 保留未被丢弃的点
            perturbed_route.append(point)
    
    return perturbed_route


def calculate_single_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    # L2归一化
    vec1_norm = F.normalize(torch.tensor(vec1), dim=0)
    vec2_norm = F.normalize(torch.tensor(vec2), dim=0)
    
    # 计算点积相似度
    similarity = torch.dot(vec1_norm, vec2_norm).item()
    return round(similarity, 4)
