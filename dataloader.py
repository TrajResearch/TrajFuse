#!/usr/bin/python
import os
import numpy as np
import torch
import warnings
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
import pickle
import pandas as pd
import json
import torch_geometric.data as geo_data
import random


def extract_time_from_list(timestamp_list):
    month_list = []
    weekday_list = []
    hour_list = []
    minute_list = []
    for timestamp in timestamp_list:
        dt = datetime.fromtimestamp(timestamp)
        month = dt.month
        weekday = dt.weekday()
        hour = dt.hour 
        minute = dt.minute 
        month_list.append(month)
        weekday_list.append(weekday)
        hour_list.append(hour)
        minute_list.append(minute)
        
        
    return torch.tensor(month_list, dtype=torch.long), torch.tensor(weekday_list, dtype=torch.long), torch.tensor(hour_list, dtype=torch.long), torch.tensor(minute_list, dtype=torch.long), torch.tensor(timestamp_list, dtype=torch.long)


#GPS轨迹经纬度数据归一化函数
def normalize_gps(lat_lists, lng_lists):
    # 收集所有轨迹的经纬度数据
    all_lats = np.concatenate(lat_lists)
    all_lngs = np.concatenate(lng_lists)

    # 计算全局极值
    lat_min, lat_max = all_lats.min(), all_lats.max()
    lng_min, lng_max = all_lngs.min(), all_lngs.max()
    print(f"GPS纬度范围: min={lat_min}, max={lat_max}")
    print(f"GPS经度范围: min={lng_min}, max={lng_max}")
    epsilon = 1e-8
    for i in range(len(lat_lists)):
        # 使用全局极值进行归一化
        if (lat_max - lat_min) > epsilon:
            lat_lists[i] = ((np.array(lat_lists[i]) - lat_min) / (lat_max - lat_min)).tolist()
        if (lng_max - lng_min) > epsilon:
            lng_lists[i] = ((np.array(lng_lists[i]) - lng_min) / (lng_max - lng_min)).tolist()

    return lat_lists, lng_lists

def prepare_time_data(timestamp_list, data_padding_value=-1, max_len=None):
    """
    准备时间戳数据
    
    Args:
        timestamp_list: 时间戳列表
        data_padding_value: 数据填充值
        max_len: 最大长度

    Returns:
        time_data: (batch, max_len, 4) - 时间特征（月份、星期、小时、分钟）
        time_mask: (batch, max_len) - 有效数据掩码
    """
    batch_size = len(timestamp_list)
    time_features_list = []
    
    for i in range(batch_size):
        timestamps = timestamp_list[i]
        
        # 提取时间特征
        month_tensor, weekday_tensor, hour_tensor, minute_tensor, timestamp_tensor = extract_time_from_list(timestamps)
        
        # 组合成时间特征矩阵 (seq_len, 5)
        seq_len = len(timestamps)
        time_features = torch.zeros((seq_len, 5), dtype=torch.long)
        time_features[:, 0] = month_tensor
        time_features[:, 1] = weekday_tensor
        time_features[:, 2] = hour_tensor
        time_features[:, 3] = minute_tensor
        time_features[:, 4] = timestamp_tensor
        
        time_features_list.append(time_features)
    
    # 如果指定了最大长度，截断序列
    if max_len is not None:
        time_features_list = [features[:max_len] for features in time_features_list]
    
    # 获取序列的最大长度
    max_seq_len = max([features.shape[0] for features in time_features_list]) if time_features_list else 0
    
    # 创建掩码
    time_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    
    # 填充到相同长度
    padded_time_data = []
    for i, features in enumerate(time_features_list):
        seq_len = features.shape[0]
        if seq_len < max_seq_len:
            padding = torch.full((max_seq_len - seq_len, 5), data_padding_value, dtype=torch.long)
            padded_features = torch.cat([features, padding], dim=0)
        else:
            padded_features = features
        padded_time_data.append(padded_features)
        time_mask[i, :seq_len] = True
    
    time_data = torch.stack(padded_time_data, dim=0)
    
    return time_data, time_mask

def prepare_gps_data(sparse_lat_list, sparse_lng_list, sparse_time_list, data_padding_value=-1.0, max_len=None):
    """
    准备GPS轨迹数据
    
    Args:
        sparse_lat_list: 稀疏GPS轨迹的纬度列表
        sparse_lng_list: 稀疏GPS轨迹的经度列表
        sparse_time_list: 稀疏GPS轨迹的时间戳列表
        data_padding_value: 数据填充值
        max_len: 最大长度

    Returns:
        gps_data: (batch, max_len, 2) - 经纬度坐标
        gps_mask: (batch, max_len) - 有效数据掩码
    """
    batch_size = len(sparse_lat_list)
    gps_coords = []

    
    for i in range(batch_size):
        lat = sparse_lat_list[i]
        lng = sparse_lng_list[i]
        
        # 确保所有列表长度相同
        min_len = min(len(lat), len(lng))
        coords = np.zeros((min_len, 2))
        
        for j in range(min_len):
            coords[j, 0] = lat[j]   # 纬度
            coords[j, 1] = lng[j]   # 经度
        
        gps_coords.append(torch.tensor(coords, dtype=torch.float32))
    
    # 如果指定了最大长度，截断序列
    if max_len is not None:
        gps_coords = [coords[:max_len] for coords in gps_coords]
    
    # 获取序列的最大长度
    max_seq_len = max([coords.shape[0] for coords in gps_coords]) if gps_coords else 0
    
    # 创建掩码
    gps_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    
    # 填充到相同长度
    padded_gps_data = []
    for i, coords in enumerate(gps_coords):
        seq_len = coords.shape[0]
        if seq_len < max_seq_len:
            padding = torch.full((max_seq_len - seq_len, 2), data_padding_value, dtype=torch.float32)
            padded_coords = torch.cat([coords, padding], dim=0)
        else:
            padded_coords = coords
        padded_gps_data.append(padded_coords)
        gps_mask[i, :seq_len] = True
    
    gps_data = torch.stack(padded_gps_data, dim=0)
    
    return gps_data, gps_mask

def prepare_cdr_data(cdr_lat_list, cdr_lng_list, cdr_time_list=None, data_padding_value=-1.0, max_len=None):
    """
    准备CDR轨迹数据
    
    Args:
        cdr_lat_list: CDR轨迹的纬度列表
        cdr_lng_list: CDR轨迹的经度列表
        cdr_time_list: CDR轨迹的时间戳列表（可选）
        data_padding_value: 数据填充值
        max_len: 最大长度

    Returns:
        cdr_data: (batch, max_len, 2) - 经纬度坐标
        cdr_mask: (batch, max_len) - 有效数据掩码
    """
    batch_size = len(cdr_lat_list)
    cdr_coords = []
    
    for i in range(batch_size):
        lat = cdr_lat_list[i]
        lng = cdr_lng_list[i]
        
        # 确保所有列表长度相同
        min_len = min(len(lat), len(lng))
        coords = np.zeros((min_len, 2))
        
        for j in range(min_len):
            coords[j, 0] = lat[j]   # 纬度
            coords[j, 1] = lng[j]   # 经度
        
        cdr_coords.append(torch.tensor(coords, dtype=torch.float32))
    
    # 如果指定了最大长度，截断序列
    if max_len is not None:
        cdr_coords = [coords[:max_len] for coords in cdr_coords]
    
    # 获取序列的最大长度
    max_seq_len = max([coords.shape[0] for coords in cdr_coords]) if cdr_coords else 0
    
    # 创建掩码
    cdr_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    
    # 填充到相同长度
    padded_cdr_data = []
    for i, coords in enumerate(cdr_coords):
        seq_len = coords.shape[0]
        if seq_len < max_seq_len:
            padding = torch.full((max_seq_len - seq_len, 2), data_padding_value, dtype=torch.float32)
            padded_coords = torch.cat([coords, padding], dim=0)
        else:
            padded_coords = coords
        padded_cdr_data.append(padded_coords)
        cdr_mask[i, :seq_len] = True
    
    cdr_data = torch.stack(padded_cdr_data, dim=0)
    
    return cdr_data, cdr_mask

def prepare_road_seg_data(road_seg_list, data_padding_value=-1, max_len=None, road_network_size=None):
    """
    准备路段序列数据
    
    Args:
        road_seg_list: 路段ID列表
        data_padding_value: 数据填充值
        max_len: 最大长度
        road_network_size: 路网大小，用于检查ID范围

    Returns:
        road_seg_data: (batch, max_len) - 路段ID序列
        road_seg_mask: (batch, max_len) - 有效数据掩码
    """
    batch_size = len(road_seg_list)
    road_segs = []
    
    for i in range(batch_size):
        segs = road_seg_list[i]
        
        # 检查并处理无效的路段ID
        if road_network_size is not None:
            valid_segs = []
            for seg in segs:
                if 0 <= seg < road_network_size:
                    valid_segs.append(seg)
                else:
                    # 无效ID替换为填充值
                    valid_segs.append(data_padding_value)
            segs = valid_segs
        
        road_segs.append(torch.tensor(segs, dtype=torch.long))
    
    # 如果指定了最大长度，截断序列
    if max_len is not None:
        road_segs = [segs[:max_len] for segs in road_segs]
    
    # 获取序列的最大长度
    max_seq_len = max([segs.shape[0] for segs in road_segs]) if road_segs else 0
    
    # 创建掩码
    road_seg_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    
    # 填充到相同长度
    padded_road_seg_data = []
    for i, segs in enumerate(road_segs):
        seq_len = segs.shape[0]
        if seq_len < max_seq_len:
            padding = torch.full((max_seq_len - seq_len,), data_padding_value, dtype=torch.long)
            padded_segs = torch.cat([segs, padding], dim=0)
        else:
            padded_segs = segs
        padded_road_seg_data.append(padded_segs)
        road_seg_mask[i, :seq_len] = True
    
    road_seg_data = torch.stack(padded_road_seg_data, dim=0)
    
    return road_seg_data, road_seg_mask

def prepare_road_network(road_segments, road_connections, segment_attrs=None):
    """
    准备路网图数据
    
    Args:
        road_segments: 路段节点列表
        road_connections: 路段连接边列表 [(start_id, end_id), ...]
        segment_attrs: 路段属性字典 {segment_id: [attrs...], ...}

    Returns:
        road_network: PyG图数据结构
    """
    # 创建边索引
    edge_index = torch.tensor(road_connections, dtype=torch.long).t().contiguous()
    
    # 创建节点特征
    if segment_attrs is None:
        # 如果没有属性，使用零向量
        num_nodes = max(road_segments) + 1
        #print("最小路段索引为：",min(road_segments))
        x = torch.zeros((num_nodes, 10), dtype=torch.float32)  # 默认10维特征
    else:
        # 使用给定的属性
        num_nodes = max(road_segments) + 1
        #print("最小路段索引为：",min(road_segments))
        x = torch.zeros((num_nodes, 10), dtype=torch.float32)
        for seg_id, attrs in segment_attrs.items():
            if seg_id < num_nodes:
                x[seg_id, :len(attrs)] = torch.tensor(attrs, dtype=torch.float32)
    
    # 创建PyG图
    road_network = geo_data.Data(x=x, edge_index=edge_index)
    
    return road_network

def random_modal_dropout(gps_data, cdr_data, road_seg_data, dropout_prob=0.5):
    """
    随机模态丢弃
    
    Args:
        gps_data: GPS数据
        cdr_data: CDR数据
        road_seg_data: 路段数据
        dropout_prob: 每个模态丢弃的概率

    Returns:
        gps_data_out: 可能为None的GPS数据
        cdr_data_out: 可能为None的CDR数据
        road_seg_data_out: 可能为None的路段数据
        modality_mask: [gps_alive, cdr_alive, seg_alive] 模态存活标记
    """
    # 生成随机数确定是否丢弃每个模态
    gps_alive = random.random() > dropout_prob
    cdr_alive = random.random() > dropout_prob
    seg_alive = random.random() > dropout_prob
    
    # 确保至少有一个模态存活
    if not (gps_alive or cdr_alive or seg_alive):
        # 随机选择一个模态保留
        modality = random.randint(0, 2)
        if modality == 0:
            gps_alive = True
        elif modality == 1:
            cdr_alive = True
        else:
            seg_alive = True
    
    # 根据存活标记返回数据
    gps_data_out = gps_data if gps_alive else None
    cdr_data_out = cdr_data if cdr_alive else None
    road_seg_data_out = road_seg_data if seg_alive else None
    modality_mask = torch.tensor([gps_alive, cdr_alive, seg_alive], dtype=torch.bool)
    
    return gps_data_out, cdr_data_out, road_seg_data_out, modality_mask

class TriFusionDataset(Dataset):
    def __init__(self, data, road_network=None, gps_max_len=None, cdr_max_len=None, route_max_len=None, 
                 modal_dropout_prob=0.0, apply_modal_dropout=False):
        """
        TriFusion数据集 - 扩展版本支持更多数据字段
        
        Args:
            data: 包含轨迹数据的DataFrame
            road_network: 路网图数据，若为None则会从data中构建
            gps_max_len: GPS轨迹最大长度
            cdr_max_len: CDR轨迹最大长度
            route_max_len: 路段序列最大长度
            modal_dropout_prob: 模态丢弃概率
            apply_modal_dropout: 是否应用模态丢弃
        """
        self.data = data
        self.modal_dropout_prob = modal_dropout_prob
        self.apply_modal_dropout = apply_modal_dropout

        # 检查并加载总时间字段
        if 'total_time' in data.columns:
            self.total_time_data = torch.tensor(data['total_time'].values, dtype=torch.float32)
            self.total_time_mean = self.total_time_data.mean().item()
            self.total_time_std = self.total_time_data.std().item()
            print(f"已加载总时间数据: {len(self.total_time_data)} 个样本")
            # 统计总时间信息
            if len(self.total_time_data) > 0:
                total_time_stats = {
                    'min': self.total_time_data.min().item(),
                    'max': self.total_time_data.max().item(),
                    'mean': self.total_time_data.mean().item(),
                    'std': self.total_time_data.std().item()
                }
                print(f"总时间统计: min={total_time_stats['min']:.2f}, max={total_time_stats['max']:.2f}, "
                      f"mean={total_time_stats['mean']:.2f}, std={total_time_stats['std']:.2f}")
            #self.total_time_data = (self.total_time_data - total_time_stats['mean']) / total_time_stats['std']
        else:
            self.total_time_data = None
            print("警告: 数据中未找到total_time字段，总时间数据将不可用")
        
        # 准备GPS轨迹数据
        sparse_lat_list = data['sparse_lat_list'].tolist()
        sparse_lng_list = data['sparse_lng_list'].tolist()
        gps_time_list = data['sparse_time_list'].tolist() if 'sparse_time_list' in data.columns else [[] for _ in range(len(sparse_lat_list))]
        
        self.gps_data, self.gps_mask = prepare_gps_data(
            sparse_lat_list, sparse_lng_list, gps_time_list,
            data_padding_value=-1.0, max_len=gps_max_len
        )
        self.gps_time_data, self.gps_time_mask = prepare_time_data(
            gps_time_list, data_padding_value=-1, max_len=gps_max_len
        )

        # 准备road_gps数据
        road_gps_data_list = data['road_gps_data'].tolist()
        road_gps_lat_list = [item[0] for item in road_gps_data_list]  # 提取纬度数据
        road_gps_lng_list = [item[1] for item in road_gps_data_list]  # 提取经度数据
        road_gps_time_list = data['road_gps_timestamp'].tolist() if 'road_gps_timestamp' in data.columns else [[] for _ in range(len(road_gps_lat_list))]
        
        self.road_gps_data, self.road_gps_mask = prepare_gps_data(
            road_gps_lat_list, road_gps_lng_list, road_gps_time_list,
            data_padding_value=-1.0, max_len=route_max_len
        )
        self.road_gps_time_data, self.road_gps_time_mask = prepare_time_data(
            road_gps_time_list, data_padding_value=-1, max_len=route_max_len
        )
        
        # 准备CDR轨迹数据
        cdr_lat_list = data['cdr_lat_list'].tolist()
        cdr_lng_list = data['cdr_lng_list'].tolist()
        cdr_time_list = data['cdr_timestamp'].tolist() if 'cdr_timestamp' in data.columns else [[] for _ in range(len(cdr_lat_list))]
        
        self.cdr_data, self.cdr_mask = prepare_cdr_data(
            cdr_lat_list, cdr_lng_list, cdr_time_list,
            data_padding_value=-1.0, max_len=cdr_max_len
        )
        self.cdr_time_data, self.cdr_time_mask = prepare_time_data(
            cdr_time_list, data_padding_value=-1, max_len=cdr_max_len
        )
        
        # 计算实际的GPS和CDR数据长度（去除padding）
        actual_gps_lengths = []
        actual_cdr_lengths = []
        actual_road_gps_lengths = []
        
        for i in range(len(data)):
            # GPS实际长度
            gps_valid_mask = self.gps_mask[i]
            actual_gps_len = gps_valid_mask.sum().item()
            actual_gps_lengths.append(actual_gps_len)
            
            # CDR实际长度  
            cdr_valid_mask = self.cdr_mask[i]
            actual_cdr_len = cdr_valid_mask.sum().item()
            actual_cdr_lengths.append(actual_cdr_len)

            # road_gps实际长度
            road_gps_valid_mask = self.road_gps_mask[i]
            actual_road_gps_len = road_gps_valid_mask.sum().item()
            actual_road_gps_lengths.append(actual_road_gps_len)
        
        # 准备路网图数据（如果没有提供）
        if road_network is None:
            # 从data中提取所有唯一路段ID
            unique_segments = set()
            
            # 添加所有路段ID字段
            for col in ['monitor_opath_list', 'sparse_gps2route_list', 'cdr_gps2route_list', 
                       'merged_route_list', 'original_cpath_list']:
                if col in data.columns:
                    road_seg_list = data[col].tolist()
                    for segs in road_seg_list:
                        if segs:  # 检查非空
                            unique_segments.update(segs)
            
            # 提取路段连接关系（如果可用）
            road_connections = []
            for col in ['monitor_opath_list', 'merged_route_list', 'original_cpath_list']:
                if col in data.columns:
                    road_seg_list = data[col].tolist()
                    for segs in road_seg_list:
                        if segs:  # 检查非空
                            for i in range(len(segs) - 1):
                                road_connections.append((segs[i], segs[i+1]))
            
            # 去重
            road_connections = list(set(road_connections))
            
            # 创建路网图
            self.road_network = prepare_road_network(
                list(unique_segments), road_connections
            )
        else:
            self.road_network = road_network
        
        # 获取路网大小，用于检查路段ID范围
        road_network_size = self.road_network.x.size(0)

        # 准备路段时间数据
        road_time_list = data['monitor_opath_timestamp'].tolist() if 'monitor_opath_timestamp' in data.columns else [[] for _ in range(len(monitor_opath_list))]
        
        self.road_time_data, self.road_time_mask = prepare_time_data(
            road_time_list, data_padding_value=-1, max_len=route_max_len
        )
        
        # 准备各种路段序列数据
        # 1. monitor_opath_list - 较为稠密路网id序列 (注意：数据中的字段名是monitor_opath_list)
        if 'monitor_opath_list' in data.columns:
            monitor_opath_list = data['monitor_opath_list'].tolist()
            self.monitor_opath_data, self.monitor_opath_mask = prepare_road_seg_data(
                monitor_opath_list, data_padding_value=-1, max_len=route_max_len,
                road_network_size=road_network_size
            )
        else:
            self.monitor_opath_data = None
            self.monitor_opath_mask = None
        
        # 2. sparse_gps2route_list - 稀疏GPS轨迹对应的路网id序列
        # 注意：使用实际GPS长度来保持与GPS数据的长度一致
        if 'sparse_gps2route_list' in data.columns:
            sparse_gps2route_list = data['sparse_gps2route_list'].tolist()
            # 对每个样本使用其实际GPS长度来截断路段数据
            truncated_sparse_gps2route_list = []
            for i, (route_list, actual_len) in enumerate(zip(sparse_gps2route_list, actual_gps_lengths)):
                if len(route_list) > actual_len:
                    truncated_route = route_list[:actual_len]
                else:
                    truncated_route = route_list
                truncated_sparse_gps2route_list.append(truncated_route)
            
            self.sparse_gps2route_data, self.sparse_gps2route_mask = prepare_road_seg_data(
                truncated_sparse_gps2route_list, data_padding_value=-1, max_len=gps_max_len,
                road_network_size=road_network_size
            )
        else:
            self.sparse_gps2route_data = None
            self.sparse_gps2route_mask = None
        
        # 3. cdr_gps2route_list - CDR轨迹对应的路网id序列
        # 注意：使用实际CDR长度来保持与CDR数据的长度一致
        if 'cdr_gps2route_list' in data.columns:
            cdr_gps2route_list = data['cdr_gps2route_list'].tolist()
            # 对每个样本使用其实际CDR长度来截断路段数据
            truncated_cdr_gps2route_list = []
            for i, (route_list, actual_len) in enumerate(zip(cdr_gps2route_list, actual_cdr_lengths)):
                if len(route_list) > actual_len:
                    truncated_route = route_list[:actual_len]
                else:
                    truncated_route = route_list
                truncated_cdr_gps2route_list.append(truncated_route)
            
            self.cdr_gps2route_data, self.cdr_gps2route_mask = prepare_road_seg_data(
                truncated_cdr_gps2route_list, data_padding_value=-1, max_len=cdr_max_len,
                road_network_size=road_network_size
            )
        else:
            self.cdr_gps2route_data = None
            self.cdr_gps2route_mask = None
        
        # 4. merged_route_list - 合并的路网id序列
        if 'merged_route_list' in data.columns:
            merged_route_list = data['merged_route_list'].tolist()
            self.merged_route_data, self.merged_route_mask = prepare_road_seg_data(
                merged_route_list, data_padding_value=-1, max_len=route_max_len,
                road_network_size=road_network_size
            )
        else:
            self.merged_route_data = None
            self.merged_route_mask = None
        
        # 5. original_cpath_list - 完整的路网id序列（作为最终target）
        if 'original_cpath_list' in data.columns:
            original_cpath_list = data['original_cpath_list'].tolist()
            self.original_cpath_data, self.original_cpath_mask = prepare_road_seg_data(
                original_cpath_list, data_padding_value=-1, max_len=route_max_len,
                road_network_size=road_network_size
            )
        else:
            # 如果没有original_cpath_list，使用monitor_opath_list作为备用
            if 'monitor_opath_list' in data.columns:
                self.original_cpath_data = self.monitor_opath_data
                self.original_cpath_mask = self.monitor_opath_mask
            else:
                print("警告：没有找到original_cpath_list或monitor_opath_list，使用空数据")
                self.original_cpath_data = torch.zeros((len(data), 1), dtype=torch.long)
                self.original_cpath_mask = torch.zeros((len(data), 1), dtype=torch.bool)
        
        # 保持向后兼容性 - 检查必要数据字段是否存在
        if self.monitor_opath_data is not None:
            self.road_seg_data = self.monitor_opath_data
            self.road_seg_mask = self.monitor_opath_mask
            print("使用monitor_opath_data作为road_seg_data")
        else:
            raise ValueError("缺少必要的数据字段：monitor_opath_list。请检查数据文件是否包含此字段。")
        
        # 检查目标数据是否存在
        if self.original_cpath_data is None:
            raise ValueError("缺少必要的数据字段：original_cpath_list。请检查数据文件是否包含此字段。")
        
        # 使用original_cpath_list作为target_seg_data
        self.target_seg_data = self.original_cpath_data
        self.target_seg_mask = self.original_cpath_mask
        
        # 添加长度一致性检查和调试信息
        print("=" * 60)
        print("数据长度一致性检查:")
        print(f"数据集大小: {len(data)}")
        
        # 检查GPS和sparse_gps2route的长度一致性
        if self.sparse_gps2route_data is not None:
            gps_route_mismatch_count = 0
            for i in range(min(len(data), 3)):  # 只检查前3个样本作为示例
                gps_len = self.gps_mask[i].sum().item()
                route_len = self.sparse_gps2route_mask[i].sum().item()
                if gps_len != route_len:
                    gps_route_mismatch_count += 1
                    print(f"样本 {i}: GPS长度={gps_len}, sparse_gps2route长度={route_len} - {'✗ 不匹配' if gps_len != route_len else '✓ 匹配'}")
                else:
                    print(f"样本 {i}: GPS长度={gps_len}, sparse_gps2route长度={route_len} - ✓ 匹配")
        
        # 检查CDR和cdr_gps2route的长度一致性
        if self.cdr_gps2route_data is not None:
            cdr_route_mismatch_count = 0
            for i in range(min(len(data), 3)):  # 只检查前3个样本作为示例
                cdr_len = self.cdr_mask[i].sum().item()
                route_len = self.cdr_gps2route_mask[i].sum().item()
                if cdr_len != route_len:
                    cdr_route_mismatch_count += 1
                    print(f"样本 {i}: CDR长度={cdr_len}, cdr_gps2route长度={route_len} - {'✗ 不匹配' if cdr_len != route_len else '✓ 匹配'}")
                else:
                    print(f"样本 {i}: CDR长度={cdr_len}, cdr_gps2route长度={route_len} - ✓ 匹配")

        # 检查CDR和cdr_time_data的长度一致性
        if self.cdr_time_data is not None:
            cdr_time_mismatch_count = 0
            for i in range(min(len(data), 3)):  # 只检查前3个样本作为示例
                cdr_len = self.cdr_mask[i].sum().item()
                time_len = self.cdr_time_mask[i].sum().item()
                if cdr_len != time_len:
                    cdr_time_mismatch_count += 1
                    print(f"样本 {i}: CDR长度={cdr_len}, cdr_time长度={time_len} - {'✗ 不匹配' if cdr_len != time_len else '✓ 匹配'}")
                else:
                    print(f"样本 {i}: CDR长度={cdr_len}, cdr_time长度={time_len} - ✓ 匹配")
        # 检查GPS和gps_time_data的长度一致性
        if self.gps_time_data is not None:
            gps_time_mismatch_count = 0
            for i in range(min(len(data), 3)):  # 只检查前3个样本作为示例
                gps_len = self.gps_mask[i].sum().item()
                time_len = self.gps_time_mask[i].sum().item()
                if gps_len != time_len:
                    gps_time_mismatch_count += 1
                    print(f"样本 {i}: GPS长度={gps_len}, gps_time长度={time_len} - {'✗ 不匹配' if gps_len != time_len else '✓ 匹配'}")
                else:
                    print(f"样本 {i}: GPS长度={gps_len}, gps_time长度={time_len} - ✓ 匹配")

        # 检查road_seg和road_time_data的长度一致性
        if self.road_time_data is not None:
            road_time_mismatch_count = 0
            for i in range(min(len(data), 3)):  # 只检查前3个样本作为示例
                road_len = self.road_seg_mask[i].sum().item()
                time_len = self.road_time_mask[i].sum().item()
                if road_len != time_len:
                    road_time_mismatch_count += 1
                    print(f"样本 {i}: road_seg长度={road_len}, road_time长度={time_len} - {'✗ 不匹配' if road_len != time_len else '✓ 匹配'}")
                else:
                    print(f"样本 {i}: road_seg长度={road_len}, road_time长度={time_len} - ✓ 匹配")

        
        print("=" * 60)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取基本数据
        gps_data = self.gps_data[idx]
        gps_mask = self.gps_mask[idx]
        cdr_data = self.cdr_data[idx]
        cdr_mask = self.cdr_mask[idx]
        road_seg_data = self.road_seg_data[idx]
        road_seg_mask = self.road_seg_mask[idx]
        target_seg_data = self.target_seg_data[idx]
        target_seg_mask = self.target_seg_mask[idx]

        # 获取总时间数据（如果可用）
        total_time = None
        if self.total_time_data is not None:
            total_time = self.total_time_data[idx]


        
        # 获取扩展数据
        item_dict = {
            'gps_data': gps_data,
            'gps_mask': gps_mask,
            'cdr_data': cdr_data,
            'cdr_mask': cdr_mask,
            'road_seg_data': road_seg_data,
            'road_seg_mask': road_seg_mask,
            'target_seg_data': target_seg_data,
            'target_seg_mask': target_seg_mask,
            'road_gps_data': self.road_gps_data[idx],
            'road_gps_mask': self.road_gps_mask[idx],
            'road_network': self.road_network,
            'total_time': total_time,
            'total_time_mean': self.total_time_mean,
            'total_time_std': self.total_time_std

        }
        
        # 添加扩展的路段数据
        if self.monitor_opath_data is not None:
            item_dict['monitor_opath_data'] = self.monitor_opath_data[idx]
            item_dict['monitor_opath_mask'] = self.monitor_opath_mask[idx]
        
        if self.sparse_gps2route_data is not None:
            item_dict['sparse_gps2route_data'] = self.sparse_gps2route_data[idx]
            item_dict['sparse_gps2route_mask'] = self.sparse_gps2route_mask[idx]
        
        if self.cdr_gps2route_data is not None:
            item_dict['cdr_gps2route_data'] = self.cdr_gps2route_data[idx]
            item_dict['cdr_gps2route_mask'] = self.cdr_gps2route_mask[idx]
        
        if self.merged_route_data is not None:
            item_dict['merged_route_data'] = self.merged_route_data[idx]
            item_dict['merged_route_mask'] = self.merged_route_mask[idx]
        
        if self.original_cpath_data is not None:
            item_dict['original_cpath_data'] = self.original_cpath_data[idx]
            item_dict['original_cpath_mask'] = self.original_cpath_mask[idx]


        # 添加扩展的时间数据
        if self.gps_time_data is not None:
            item_dict['gps_time_data'] = self.gps_time_data[idx]
            item_dict['gps_time_mask'] = self.gps_time_mask[idx]
        
        if self.cdr_time_data is not None:
            item_dict['cdr_time_data'] = self.cdr_time_data[idx]
            item_dict['cdr_time_mask'] = self.cdr_time_mask[idx]
        
        if self.road_time_data is not None:
            item_dict['road_time_data'] = self.road_time_data[idx]
            item_dict['road_time_mask'] = self.road_time_mask[idx]

        if self.road_gps_time_data is not None:
            item_dict['road_gps_time_data'] = self.road_gps_time_data[idx]
            item_dict['road_gps_time_mask'] = self.road_gps_time_mask[idx]
        
        # 应用模态丢弃（如果需要）
        if self.apply_modal_dropout:
            # 注意：这里传入的是单个样本，所以我们需要为random_modal_dropout添加批次维度
            gps_data_batch = gps_data.unsqueeze(0)
            cdr_data_batch = cdr_data.unsqueeze(0)
            road_seg_data_batch = road_seg_data.unsqueeze(0)
            
            gps_data_out, cdr_data_out, road_seg_data_out, modality_mask = random_modal_dropout(
                gps_data_batch, cdr_data_batch, road_seg_data_batch, self.modal_dropout_prob
            )
            
            # 去除批次维度
            if gps_data_out is not None:
                item_dict['gps_data'] = gps_data_out.squeeze(0)
            else:
                item_dict['gps_data'] = None
            if cdr_data_out is not None:
                item_dict['cdr_data'] = cdr_data_out.squeeze(0)
            else:
                item_dict['cdr_data'] = None
            if road_seg_data_out is not None:
                item_dict['road_seg_data'] = road_seg_data_out.squeeze(0)
            else:
                item_dict['road_seg_data'] = None
                
            item_dict['modality_mask'] = modality_mask
        else:
            item_dict['modality_mask'] = torch.tensor([True, True, True], dtype=torch.bool)
        
        return item_dict

def collate_with_modality_dropout(batch):
    """
    批次整理函数，处理可能为None的模态数据以及扩展的数据字段
    
    Args:
        batch: 数据批次列表

    Returns:
        batch_dict: 整理后的批次字典
    """
    batch_size = len(batch)
    
    # 检查每个模态是否存在（至少有一个样本包含该模态）
    gps_exists = any(item['gps_data'] is not None for item in batch)
    cdr_exists = any(item['cdr_data'] is not None for item in batch)
    seg_exists = any(item['road_seg_data'] is not None for item in batch)
    road_gps_exists = any(item['road_gps_data'] is not None for item in batch)

    # 检查总时间数据是否存在
    total_time_exists = any(item.get('total_time') is not None for item in batch)
    
    # 初始化返回字典
    batch_dict = {
        'gps_data': None,
        'gps_mask': None,
        'cdr_data': None,
        'cdr_mask': None,
        'road_seg_data': None,
        'road_seg_mask': None,
        'road_gps_data': None,
        'road_gps_mask': None,
        'target_seg_data': None,
        'target_seg_mask': None,
        'modality_mask': None,
        'road_network': None,
        # 扩展字段
        'monitor_opath_data': None,
        'monitor_opath_mask': None,
        'sparse_gps2route_data': None,
        'sparse_gps2route_mask': None,
        'cdr_gps2route_data': None,
        'cdr_gps2route_mask': None,
        'merged_route_data': None,
        'merged_route_mask': None,
        'original_cpath_data': None,
        'original_cpath_mask': None,
        # 扩展时间字段
        'gps_time_data': None,
        'gps_time_mask': None,
        'cdr_time_data': None,
        'cdr_time_mask': None,
        'road_time_data': None,
        'road_time_mask': None,
        'total_time': None,
        'total_time_mean': None,
        'total_time_std': None
    }
    
    # 整合模态存活标记
    modality_masks = [item['modality_mask'] for item in batch]
    batch_dict['modality_mask'] = torch.stack(modality_masks, dim=0)

    # 处理总时间数据
    if total_time_exists:
        total_times = []
        for item in batch:
            if item.get('total_time') is not None:
                total_times.append(item['total_time'])
            else:
                # 如果某个样本没有总时间，填充默认值（0）
                total_times.append(torch.tensor(0.0, dtype=torch.float32))
        batch_dict['total_time'] = torch.stack(total_times, dim=0)
        # 获取全局统计量（从第一个样本获取，因为所有样本应该相同）
        if 'total_time_mean' in batch[0] and 'total_time_std' in batch[0]:
            batch_dict['total_time_mean'] = batch[0]['total_time_mean']  # 单个值
            batch_dict['total_time_std'] = batch[0]['total_time_std']    # 单个值

    # 处理GPS数据
    if gps_exists:
        gps_data = []
        gps_mask = []
        max_gps_len = 0
        
        # 先收集非None的数据以确定最大长度
        for item in batch:
            if item['gps_data'] is not None:
                max_gps_len = max(max_gps_len, item['gps_data'].size(0))
        
        # 为每个样本准备GPS数据（如果为None则填充零）
        for item in batch:
            if item['gps_data'] is not None:
                gps_data.append(item['gps_data'])
                gps_mask.append(item['gps_mask'])
            else:
                # 填充默认的GPS数据
                default_gps = torch.zeros(max_gps_len, 2, dtype=torch.float32)
                default_mask = torch.zeros(max_gps_len, dtype=torch.bool)
                gps_data.append(default_gps)
                gps_mask.append(default_mask)
        
        # 填充到相同长度
        max_len = max([data.size(0) for data in gps_data])
        padded_gps_data = []
        padded_gps_mask = []
        for data, mask in zip(gps_data, gps_mask):
            if data.size(0) < max_len:
                padding = torch.zeros(max_len - data.size(0), 2, dtype=torch.float32)
                data = torch.cat([data, padding], dim=0)
                mask_padding = torch.zeros(max_len - mask.size(0), dtype=torch.bool)
                mask = torch.cat([mask, mask_padding], dim=0)
            padded_gps_data.append(data)
            padded_gps_mask.append(mask)
        
        batch_dict['gps_data'] = torch.stack(padded_gps_data, dim=0)
        batch_dict['gps_mask'] = torch.stack(padded_gps_mask, dim=0)
    
    # 处理路网GPS数据
    if road_gps_exists:
        road_gps_data = []
        road_gps_mask = []
        max_road_gps_len = 0
        
        # 先收集非None的数据以确定最大长度
        for item in batch:
            if item['road_gps_data'] is not None:
                max_road_gps_len = max(max_road_gps_len, item['road_gps_data'].size(0))
        
        # 为每个样本准备路网GPS数据（如果为None则填充零）
        for item in batch:
            if item['road_gps_data'] is not None:
                road_gps_data.append(item['road_gps_data'])
                road_gps_mask.append(item['road_gps_mask'])
            else:
                # 填充默认的路网GPS数据
                default_road_gps = torch.zeros(max_road_gps_len, 2, dtype=torch.float32)
                default_mask = torch.zeros(max_road_gps_len, dtype=torch.bool)
                road_gps_data.append(default_road_gps)
                road_gps_mask.append(default_mask)
        
        # 填充到相同长度
        max_len = max([data.size(0) for data in road_gps_data])
        padded_road_gps_data = []
        padded_road_gps_mask = []
        for data, mask in zip(road_gps_data, road_gps_mask):
            if data.size(0) < max_len:
                padding = torch.zeros(max_len - data.size(0), 2, dtype=torch.float32)
                data = torch.cat([data, padding], dim=0)
                mask_padding = torch.zeros(max_len - mask.size(0), dtype=torch.bool)
                mask = torch.cat([mask, mask_padding], dim=0)
            padded_road_gps_data.append(data)
            padded_road_gps_mask.append(mask)
        
        batch_dict['road_gps_data'] = torch.stack(padded_road_gps_data, dim=0)
        batch_dict['road_gps_mask'] = torch.stack(padded_road_gps_mask, dim=0)

    # 处理CDR数据
    if cdr_exists:
        cdr_data = []
        cdr_mask = []
        max_cdr_len = 0
        
        # 先收集非None的数据以确定最大长度
        for item in batch:
            if item['cdr_data'] is not None:
                max_cdr_len = max(max_cdr_len, item['cdr_data'].size(0))
        
        # 为每个样本准备CDR数据（如果为None则填充零）
        for item in batch:
            if item['cdr_data'] is not None:
                cdr_data.append(item['cdr_data'])
                cdr_mask.append(item['cdr_mask'])
            else:
                # 填充默认的CDR数据
                default_cdr = torch.zeros(max_cdr_len, 2, dtype=torch.float32)
                default_mask = torch.zeros(max_cdr_len, dtype=torch.bool)
                cdr_data.append(default_cdr)
                cdr_mask.append(default_mask)
        
        # 填充到相同长度
        max_len = max([data.size(0) for data in cdr_data])
        padded_cdr_data = []
        padded_cdr_mask = []
        for data, mask in zip(cdr_data, cdr_mask):
            if data.size(0) < max_len:
                padding = torch.zeros(max_len - data.size(0), 2, dtype=torch.float32)
                data = torch.cat([data, padding], dim=0)
                mask_padding = torch.zeros(max_len - mask.size(0), dtype=torch.bool)
                mask = torch.cat([mask, mask_padding], dim=0)
            padded_cdr_data.append(data)
            padded_cdr_mask.append(mask)
        
        batch_dict['cdr_data'] = torch.stack(padded_cdr_data, dim=0)
        batch_dict['cdr_mask'] = torch.stack(padded_cdr_mask, dim=0)
    
    # 处理路段数据
    if seg_exists:
        road_seg_data = []
        road_seg_mask = []
        max_seg_len = 0
        
        # 先收集非None的数据以确定最大长度
        for item in batch:
            if item['road_seg_data'] is not None:
                max_seg_len = max(max_seg_len, item['road_seg_data'].size(0))
        
        # 为每个样本准备路段数据（如果为None则填充零）
        for item in batch:
            if item['road_seg_data'] is not None:
                road_seg_data.append(item['road_seg_data'])
                road_seg_mask.append(item['road_seg_mask'])
            else:
                # 填充默认的路段数据
                default_seg = torch.zeros(max_seg_len, dtype=torch.long)
                default_mask = torch.zeros(max_seg_len, dtype=torch.bool)
                road_seg_data.append(default_seg)
                road_seg_mask.append(default_mask)
        
        # 填充到相同长度
        max_len = max([data.size(0) for data in road_seg_data])
        padded_seg_data = []
        padded_seg_mask = []
        for data, mask in zip(road_seg_data, road_seg_mask):
            if data.size(0) < max_len:
                padding = torch.zeros(max_len - data.size(0), dtype=torch.long)
                data = torch.cat([data, padding], dim=0)
                mask_padding = torch.zeros(max_len - mask.size(0), dtype=torch.bool)
                mask = torch.cat([mask, mask_padding], dim=0)
            padded_seg_data.append(data)
            padded_seg_mask.append(mask)
        
        batch_dict['road_seg_data'] = torch.stack(padded_seg_data, dim=0)
        batch_dict['road_seg_mask'] = torch.stack(padded_seg_mask, dim=0)
    

    # 处理目标路段数据（始终存在）
    target_seg_data = [item['target_seg_data'] for item in batch]
    target_seg_mask = [item['target_seg_mask'] for item in batch]
    batch_dict['target_seg_data'] = torch.stack(target_seg_data, dim=0)
    batch_dict['target_seg_mask'] = torch.stack(target_seg_mask, dim=0)
    
    # 处理扩展的路段数据字段
    def _collate_optional_field(field_name):
        if any(field_name in item for item in batch):
            field_data = [item.get(field_name) for item in batch]
            if any(data is not None for data in field_data):
                # 过滤掉None值并填充
                valid_data = [data for data in field_data if data is not None]
                if valid_data:
                    return torch.stack(valid_data, dim=0)
        return None
    
    # 处理各种扩展数据字段
    extended_fields = [
        'monitor_opath_data', 'monitor_opath_mask',
        'sparse_gps2route_data', 'sparse_gps2route_mask', 
        'cdr_gps2route_data', 'cdr_gps2route_mask',
        'merged_route_data', 'merged_route_mask',
        'original_cpath_data', 'original_cpath_mask',
        # 扩展时间字段
        'gps_time_data', 'gps_time_mask',
        'cdr_time_data', 'cdr_time_mask',
        'road_time_data', 'road_time_mask',
        'road_gps_time_data', 'road_gps_time_mask'
    ]
    
    for field in extended_fields:
        if any(field in item for item in batch):
            field_data = []
            for item in batch:
                if field in item and item[field] is not None:
                    field_data.append(item[field])
            
            if field_data:
                # 填充到相同长度
                max_len = max([data.size(0) for data in field_data])
                padded_field_data = []
                
                for data in field_data:
                    if data.size(0) < max_len:
                        if 'mask' in field:
                            padding = torch.zeros(max_len - data.size(0), dtype=torch.bool)
                        else:
                            padding = torch.zeros(max_len - data.size(0), dtype=torch.long)
                        data = torch.cat([data, padding], dim=0)
                    padded_field_data.append(data)
                
                batch_dict[field] = torch.stack(padded_field_data, dim=0)
    
    # 路网数据是共享的，使用第一个样本的
    batch_dict['road_network'] = batch[0]['road_network']
    
    return batch_dict

def get_trifusion_loaders(
    data_path, 
    batch_size, 
    num_workers,
    route_min_len=5, 
    route_max_len=100, 
    gps_min_len=5, 
    gps_max_len=1000,
    cdr_min_len=3, 
    cdr_max_len=100,
    num_samples=10000, 
    test_ratio=0.05,
    modal_dropout_prob=0.0,  # 修改默认值为0.0，确保默认不丢弃模态
    apply_modal_dropout_train=False,
    apply_modal_dropout_test=False,
    seed=42
):
    """
    获取TriFusion训练和测试数据加载器
    
    Args:
        data_path: 数据路径
        batch_size: 批次大小
        num_workers: 工作线程数
        route_min_len: 最小路段长度
        route_max_len: 最大路段长度
        gps_min_len: 最小GPS轨迹长度
        gps_max_len: 最大GPS轨迹长度
        cdr_min_len: 最小CDR轨迹长度
        cdr_max_len: 最大CDR轨迹长度
        num_samples: 样本数量
        test_ratio: 测试集比例
        modal_dropout_prob: 模态丢弃概率（默认0.0表示不丢弃任何模态）
        apply_modal_dropout_train: 训练集是否应用模态丢弃（默认False）
        apply_modal_dropout_test: 测试集是否应用模态丢弃（默认False）
        seed: 随机种子
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        road_network: 路网图数据
        normalization_stats: 标准化统计信息
    """
    # 设置随机种子 - 确保完整的随机性控制
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保cudnn的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("=" * 60)
    print("DataLoader 确定性配置")
    print("=" * 60)
    print(f"随机种子: {seed}")
    print(f"NumPy种子: {seed}")
    print(f"Python random种子: {seed}")
    print(f"PyTorch种子: {seed}")
    print(f"CUDA种子控制: 已启用")
    print(f"cuDNN确定性: True")
    print(f"cuDNN基准测试: False")
    print("=" * 60)
    
    # 创建确定性的随机数生成器
    generator = torch.Generator()
    generator.manual_seed(seed)
    print(f"DataLoader generator种子: {seed}")
    
    # 定义worker初始化函数
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        print(f"Worker {worker_id} 种子: {worker_seed}")
    
    # 加载数据
    dataset = pd.read_pickle(data_path)
    print(f"原始数据: {dataset.shape}")
    
    # 过滤轨迹长度
    dataset = dataset[
        dataset['monitor_opath_list'].apply(len) > route_min_len
    ].reset_index(drop=True)
    
    dataset = dataset[
        dataset['monitor_opath_list'].apply(len) < route_max_len
    ].reset_index(drop=True)
    
    dataset = dataset[
        dataset['sparse_lat_list'].apply(len) > gps_min_len
    ].reset_index(drop=True)
    
    dataset = dataset[
        dataset['sparse_lat_list'].apply(len) < gps_max_len
    ].reset_index(drop=True)
    
    dataset = dataset[
        dataset['cdr_lat_list'].apply(len) > cdr_min_len
    ].reset_index(drop=True)
    
    dataset = dataset[
        dataset['cdr_lat_list'].apply(len) < cdr_max_len
    ].reset_index(drop=True)
    
    print("=" * 50)
    print("数据过滤和采样详情")
    print("=" * 50)
    print(f"过滤后数据: {dataset.shape}")
    print(f"数据过滤条件:")
    print(f"  - 路段长度: {route_min_len} < len < {route_max_len}")
    print(f"  - GPS轨迹长度: {gps_min_len} < len < {gps_max_len}")
    print(f"  - CDR轨迹长度: {cdr_min_len} < len < {cdr_max_len}")

    
    # 确保样本数量不超过可用数据
    if num_samples is None:
        num_samples = dataset.shape[0]
        print(f"使用全部 {num_samples} 个样本")
    elif dataset.shape[0] < num_samples:
        num_samples = dataset.shape[0]
        print(f"警告: 可用样本数量不足，使用全部 {num_samples} 个样本")
    else:
        print(f"采样 {num_samples} 个样本 (共 {dataset.shape[0]} 个可用)")
    
    # 随机采样
    print(f"随机采样种子: {seed}")
    sampled_data = dataset.sample(n=num_samples, replace=False, random_state=seed)
    print(f"采样完成: {sampled_data.shape}")
    print("=" * 50)
    
    # 计算标准化统计信息（基于采样后的数据）
    normalization_stats = compute_normalization_stats(
        sampled_data, 
        gps_max_len=gps_max_len, 
        cdr_max_len=cdr_max_len
    )
    
    # 准备路网图数据
    road_seg_list = sampled_data['monitor_opath_list'].tolist()
    
    # 从data中提取所有唯一路段ID
    unique_segments = set()
    for segs in road_seg_list:
        unique_segments.update(segs)
    
    # 提取路段连接关系
    road_connections = []
    for segs in road_seg_list:
        for i in range(len(segs) - 1):
            road_connections.append((segs[i], segs[i+1]))
    
    # 去重
    road_connections = list(set(road_connections))
    
    # 创建路网图
    road_network = prepare_road_network(
        list(unique_segments), road_connections
    )
    
    # 打印路网大小
    print(f"路网大小: {road_network.x.size(0)} 个节点, {road_network.edge_index.size(1)} 条边")
    
    # 检查是否有超出路网大小的路段ID
    max_seg_id = max(unique_segments)
    if max_seg_id >= road_network.x.size(0):
        print(f"警告: 存在超出路网大小的路段ID，最大ID: {max_seg_id}，路网大小: {road_network.x.size(0)}")
    
    # 分割训练集和测试集
    test_size = int(num_samples * test_ratio)
    train_size = num_samples - test_size
    
    # 特殊处理：当test_ratio接近1.0时，使用全部数据作为测试集，少量数据作为训练集
    if test_ratio >= 0.99:
        print(f"检测到test_ratio={test_ratio}，使用推理模式：全部数据作为测试集，少量数据作为训练集以避免空数据集错误")
        test_data = sampled_data  # 全部数据作为测试集
        train_data = sampled_data.iloc[:min(1, num_samples)]  # 取第一个样本作为训练集
        test_size = num_samples
        train_size = min(1, num_samples)
    else:
        train_data = sampled_data.iloc[:train_size]
        test_data = sampled_data.iloc[train_size:]
    
    print(f"训练集大小: {train_size}, 测试集大小: {test_size}")
    
    # 创建数据集
    train_dataset = TriFusionDataset(
        train_data,
        road_network=road_network,
        gps_max_len=gps_max_len,
        cdr_max_len=cdr_max_len,
        route_max_len=route_max_len,
        modal_dropout_prob=modal_dropout_prob,
        apply_modal_dropout=apply_modal_dropout_train
    )
    
    test_dataset = TriFusionDataset(
        test_data,
        road_network=road_network,
        gps_max_len=gps_max_len,
        cdr_max_len=cdr_max_len,
        route_max_len=route_max_len,
        modal_dropout_prob=modal_dropout_prob,
        apply_modal_dropout=apply_modal_dropout_test
    )
    
    # 创建数据加载器
    print("=" * 60)
    print("DataLoader 创建配置")
    print("=" * 60)
    print(f"批次大小: {batch_size}")
    print(f"工作线程数: {num_workers}")
    print(f"训练数据shuffle: True")
    print(f"测试数据shuffle: False")
    print(f"pin_memory: False")
    print(f"确定性generator: 已设置")
    print(f"worker_init_fn: {'已设置' if num_workers > 0 else '未使用 (单线程)'}")
    print(f"persistent_workers: {True if num_workers > 0 else False}")
    print(f"模态丢弃概率: {modal_dropout_prob}")
    print(f"训练时应用模态丢弃: {apply_modal_dropout_train}")
    print(f"测试时应用模态丢弃: {apply_modal_dropout_test}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # 关闭pin_memory，避免对CUDA张量执行pin_memory操作
        collate_fn=collate_with_modality_dropout,
        generator=generator,  # 确定性随机数生成器
        worker_init_fn=worker_init_fn if num_workers > 0 else None,  # 多进程种子控制
        persistent_workers=True if num_workers > 0 else False  # 避免重复初始化
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # 关闭pin_memory，避免对CUDA张量执行pin_memory操作
        collate_fn=collate_with_modality_dropout,
        generator=generator,  # 确定性随机数生成器（虽然不shuffle但保持一致性）
        worker_init_fn=worker_init_fn if num_workers > 0 else None,  # 多进程种子控制
        persistent_workers=True if num_workers > 0 else False  # 避免重复初始化
    )
    
    print(f"训练DataLoader创建完成: {len(train_loader)} 批次")
    print(f"测试DataLoader创建完成: {len(test_loader)} 批次")
    print("=" * 60)
    
    return train_loader, test_loader, road_network, normalization_stats

def compute_normalization_stats(data, gps_max_len=None, cdr_max_len=None):
    """
    计算GPS和CDR数据的标准化统计信息
    
    Args:
        data: DataFrame包含轨迹数据
        gps_max_len: GPS轨迹最大长度
        cdr_max_len: CDR轨迹最大长度
        
    Returns:
        stats_dict: 包含标准化参数的字典
    """
    print("正在计算GPS和CDR数据的标准化统计信息...")
    
    # 收集所有GPS坐标
    all_gps_coords = []
    for i in range(len(data)):
        lat_list = data.iloc[i]['sparse_lat_list']
        lng_list = data.iloc[i]['sparse_lng_list']
        
        min_len = min(len(lat_list), len(lng_list))
        if gps_max_len is not None:
            min_len = min(min_len, gps_max_len)
            
        for j in range(min_len):
            if -180 <= lng_list[j] <= 180 and -90 <= lat_list[j] <= 90:  # 有效坐标范围
                all_gps_coords.append([lng_list[j], lat_list[j]])  # [经度, 纬度]
    
    # 收集所有road_gps坐标
    all_road_gps_coords = []
    for i in range(len(data)):
        lat_list = data.iloc[i]['road_gps_data'][0]
        lng_list = data.iloc[i]['road_gps_data'][1]
        
        min_len = min(len(lat_list), len(lng_list))
        if gps_max_len is not None:
            min_len = min(min_len, gps_max_len)
            
        for j in range(min_len):
            if -180 <= lng_list[j] <= 180 and -90 <= lat_list[j] <= 90:  # 有效坐标范围
                all_road_gps_coords.append([lng_list[j], lat_list[j]])  # [经度, 纬度]

    # 收集所有CDR坐标
    all_cdr_coords = []
    for i in range(len(data)):
        lat_list = data.iloc[i]['cdr_lat_list']
        lng_list = data.iloc[i]['cdr_lng_list']
        
        min_len = min(len(lat_list), len(lng_list))
        if cdr_max_len is not None:
            min_len = min(min_len, cdr_max_len)
            
        for j in range(min_len):
            if -180 <= lng_list[j] <= 180 and -90 <= lat_list[j] <= 90:  # 有效坐标范围
                all_cdr_coords.append([lng_list[j], lat_list[j]])  # [经度, 纬度]
    
    # 计算GPS统计量
    if all_gps_coords:
        gps_coords_array = np.array(all_gps_coords)
        gps_mean = np.mean(gps_coords_array, axis=0)  # [mean_lon, mean_lat]
        gps_std = np.std(gps_coords_array, axis=0)    # [std_lon, std_lat]
        gps_min = np.min(gps_coords_array, axis=0)    # [min_lon, min_lat]
        gps_max = np.max(gps_coords_array, axis=0)    # [max_lon, max_lat]
        
        # 避免标准差为0
        gps_std = np.maximum(gps_std, 1e-6)
        
        print(f"GPS统计信息: 样本数={len(all_gps_coords)}")
        print(f"  经度: min={gps_min[0]:.6f}, max={gps_max[0]:.6f}, mean={gps_mean[0]:.6f}, std={gps_std[0]:.6f}")
        print(f"  纬度: min={gps_min[1]:.6f}, max={gps_max[1]:.6f}, mean={gps_mean[1]:.6f}, std={gps_std[1]:.6f}")
    else:
        print("警告: 没有找到有效的GPS数据")
        gps_mean = np.array([104.1, 30.65])  # 默认值
        gps_std = np.array([0.3, 0.15])      # 默认值
        gps_min = np.array([-180.0, -90.0])  # 默认最小值
        gps_max = np.array([180.0, 90.0])    # 默认最大值

    # 计算road_gps统计量
    if all_road_gps_coords:
        road_gps_coords_array = np.array(all_road_gps_coords)
        road_gps_mean = np.mean(road_gps_coords_array, axis=0)  # [mean_lon, mean_lat]
        road_gps_std = np.std(road_gps_coords_array, axis=0)    # [std_lon, std_lat]
        road_gps_min = np.min(road_gps_coords_array, axis=0)    # [min_lon, min_lat]
        road_gps_max = np.max(road_gps_coords_array, axis=0)    # [max_lon, max_lat]
        
        # 避免标准差为0
        road_gps_std = np.maximum(road_gps_std, 1e-6)
        
        print(f"road_gps统计信息: 样本数={len(all_road_gps_coords)}")
        print(f"  经度: min={road_gps_min[0]:.6f}, max={road_gps_max[0]:.6f}, mean={road_gps_mean[0]:.6f}, std={road_gps_std[0]:.6f}")
        print(f"  纬度: min={road_gps_min[1]:.6f}, max={road_gps_max[1]:.6f}, mean={road_gps_mean[1]:.6f}, std={road_gps_std[1]:.6f}")
    else:
        print("警告: 没有找到有效的road_gps数据")
        road_gps_mean = np.array([104.1, 30.65])  # 默认值
        road_gps_std = np.array([0.3, 0.15])      # 默认值
        road_gps_min = np.array([-180.0, -90.0])  # 默认最小值
        road_gps_max = np.array([180.0, 90.0])    # 默认最大值
    
    # 计算CDR统计量
    if all_cdr_coords:
        cdr_coords_array = np.array(all_cdr_coords)
        cdr_mean = np.mean(cdr_coords_array, axis=0)  # [mean_lon, mean_lat]
        cdr_std = np.std(cdr_coords_array, axis=0)    # [std_lon, std_lat]
        cdr_min = np.min(cdr_coords_array, axis=0)    # [min_lon, min_lat]
        cdr_max = np.max(cdr_coords_array, axis=0)    # [max_lon, max_lat]
        
        # 避免标准差为0
        cdr_std = np.maximum(cdr_std, 1e-6)
        
        print(f"CDR统计信息: 样本数={len(all_cdr_coords)}")
        print(f"  经度: min={cdr_min[0]:.6f}, max={cdr_max[0]:.6f}, mean={cdr_mean[0]:.6f}, std={cdr_std[0]:.6f}")
        print(f"  纬度: min={cdr_min[1]:.6f}, max={cdr_max[1]:.6f}, mean={cdr_mean[1]:.6f}, std={cdr_std[1]:.6f}")
    else:
        print("警告: 没有找到有效的CDR数据")
        cdr_mean = np.array([104.1, 30.65])  # 默认值
        cdr_std = np.array([0.3, 0.15])      # 默认值
        cdr_min = np.array([-180.0, -90.0])  # 默认最小值
        cdr_max = np.array([180.0, 90.0])    # 默认最大值
    
    stats_dict = {
        'gps_mean_lon': float(gps_mean[0]),
        'gps_mean_lat': float(gps_mean[1]),
        'gps_std_lon': float(gps_std[0]),
        'gps_std_lat': float(gps_std[1]),
        'gps_min_lon': float(gps_min[0]),
        'gps_max_lon': float(gps_max[0]),
        'gps_min_lat': float(gps_min[1]),
        'gps_max_lat': float(gps_max[1]),
        'cdr_mean_lon': float(cdr_mean[0]),
        'cdr_mean_lat': float(cdr_mean[1]),
        'cdr_std_lon': float(cdr_std[0]),
        'cdr_std_lat': float(cdr_std[1]),
        'cdr_min_lon': float(cdr_min[0]),
        'cdr_max_lon': float(cdr_max[0]),
        'cdr_min_lat': float(cdr_min[1]),
        'cdr_max_lat': float(cdr_max[1]),   
        'road_gps_mean_lon': float(road_gps_mean[0]),
        'road_gps_mean_lat': float(road_gps_mean[1]),
        'road_gps_std_lon': float(road_gps_std[0]),
        'road_gps_std_lat': float(road_gps_std[1]),
        'road_gps_min_lon': float(road_gps_min[0]),
        'road_gps_max_lon': float(road_gps_max[0]),
        'road_gps_min_lat': float(road_gps_min[1]),
        'road_gps_max_lat': float(road_gps_max[1]),
        'gps_sample_count': len(all_road_gps_coords),
        'cdr_sample_count': len(all_cdr_coords),
        'road_gps_sample_count': len(all_road_gps_coords)
    }
    
    print("标准化统计信息计算完成")
    return stats_dict
