#!/usr/bin/python
import os
import sys
import argparse
import numpy as np
import pandas as pd

# 将项目根目录加入 sys.path，允许从项目根的包进行绝对导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import json
import pickle
from tqdm import tqdm
import logging
from datetime import datetime
import faiss
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 设置CUDA设备
dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
if torch.cuda.is_available():
    torch.cuda.set_device(dev_id)

# 导入项目模块
from model.TrajFuse import TrajFuse
from dataloader import get_trifusion_loaders
from metrics_simple import cal_pre_recall_with_f1
from eval_utils import next_batch_index,label_norm,MLPReg,pred_unnorm
from eval_utils import DestinationPredictor as DP
from task.similarity_eval_new import search_and_evaluate



def split_gps_data_by_odd_even(gps_data):
    """
    将GPS数据的奇数位和偶数位分开，生成连续的数据，不包含间隔的填充位
    
    Args:
        gps_data: torch.Tensor, 形状为 [batch_size, seq_len, 2] 的GPS数据
        
    Returns:
        odd_data: torch.Tensor, 奇数位数据，形状为 [batch_size, seq_len, 2]
        even_data: torch.Tensor, 偶数位数据，形状为 [batch_size, seq_len, 2]
    """
    batch_size, seq_len, _ = gps_data.shape
    org_gps_data = gps_data.clone()
    
    # 创建奇数位掩码和偶数位掩码
    indices = torch.arange(seq_len, device=gps_data.device)
    odd_mask = (indices % 2 == 1)  # 奇数索引
    even_mask = (indices % 2 == 0)  # 偶数索引
    
    # 扩展掩码到batch维度
    odd_mask_expanded = odd_mask.unsqueeze(0).expand(batch_size, -1)
    even_mask_expanded = even_mask.unsqueeze(0).expand(batch_size, -1)
    
    # 提取非填充的奇数位和偶数位数据
    odd_non_padding = gps_data[odd_mask_expanded].reshape(batch_size, -1, 2)
    even_non_padding = gps_data[even_mask_expanded].reshape(batch_size, -1, 2)
    
    # 计算每个batch中奇数位和偶数位的有效长度
    odd_valid_lengths = torch.sum(~((gps_data[:, :, 0] == -1.0) & (gps_data[:, :, 1] == -1.0)) & odd_mask_expanded, dim=1)
    even_valid_lengths = torch.sum(~((gps_data[:, :, 0] == -1.0) & (gps_data[:, :, 1] == -1.0)) & even_mask_expanded, dim=1)
    
    # 初始化奇数位和偶数位数据，全部设为填充值-1
    odd_data = torch.full_like(gps_data, -1.0)
    even_data = torch.full_like(gps_data, -1.0)
    
    # 将连续的数据填充到结果中
    for i in range(batch_size):
        # 奇数位数据
        if odd_valid_lengths[i] > 0:
            odd_data[i, :odd_valid_lengths[i]] = odd_non_padding[i, :odd_valid_lengths[i]]
        
        # 偶数位数据
        if even_valid_lengths[i] > 0:
            even_data[i, :even_valid_lengths[i]] = even_non_padding[i, :even_valid_lengths[i]]


    return odd_data, even_data, org_gps_data

def split_grid_data_by_odd_even(grid_data):
    """
    将网格数据的奇数位和偶数位分开，生成连续的数据，不包含间隔的填充位
    
    Args:
        grid_data: torch.Tensor, 形状为 [batch_size, seq_len] 的网格数据
        
    Returns:
        odd_data: torch.Tensor, 奇数位数据，形状为 [batch_size, seq_len]
        even_data: torch.Tensor, 偶数位数据，形状为 [batch_size, seq_len]
    """
    batch_size, seq_len = grid_data.shape
    org_grid_data = grid_data.clone()
    
    # 创建奇数位掩码和偶数位掩码
    indices = torch.arange(seq_len, device=grid_data.device)
    odd_mask = (indices % 2 == 1)  # 奇数索引
    even_mask = (indices % 2 == 0)  # 偶数索引
    
    # 扩展掩码到batch维度
    odd_mask_expanded = odd_mask.unsqueeze(0).expand(batch_size, -1)
    even_mask_expanded = even_mask.unsqueeze(0).expand(batch_size, -1)
    
    # 提取非填充的奇数位和偶数位数据
    odd_non_padding = grid_data[odd_mask_expanded].reshape(batch_size, -1)
    even_non_padding = grid_data[even_mask_expanded].reshape(batch_size, -1)
    
    # 计算每个batch中奇数位和偶数位的有效长度（网格数据的填充值通常为-1）
    odd_valid_lengths = torch.sum((grid_data != -1) & odd_mask_expanded, dim=1)
    even_valid_lengths = torch.sum((grid_data != -1) & even_mask_expanded, dim=1)
    
    # 初始化奇数位和偶数位数据，全部设为填充值-1
    odd_data = torch.full_like(grid_data, -1)
    even_data = torch.full_like(grid_data, -1)
    
    # 将连续的数据填充到结果中
    for i in range(batch_size):
        # 奇数位数据
        if odd_valid_lengths[i] > 0:
            odd_data[i, :odd_valid_lengths[i]] = odd_non_padding[i, :odd_valid_lengths[i]]
        
        # 偶数位数据
        if even_valid_lengths[i] > 0:
            even_data[i, :even_valid_lengths[i]] = even_non_padding[i, :even_valid_lengths[i]]
    
    return odd_data, even_data, org_grid_data

def split_time_data_by_odd_even(time_data):
    """
    将时间数据的奇数位和偶数位分开，生成连续的数据，不包含间隔的填充位
    
    Args:
        time_data: torch.Tensor, 形状为 [batch_size, seq_len, 5] 的时间数据
        
    Returns:
        odd_data: torch.Tensor, 奇数位数据，形状为 [batch_size, seq_len, 5]
        even_data: torch.Tensor, 偶数位数据，形状为 [batch_size, seq_len, 5]
    """
    batch_size, seq_len, time_dim = time_data.shape
    org_time_data = time_data.clone()
    
    # 创建奇数位掩码和偶数位掩码
    indices = torch.arange(seq_len, device=time_data.device)
    odd_mask = (indices % 2 == 1)  # 奇数索引
    even_mask = (indices % 2 == 0)  # 偶数索引
    
    # 扩展掩码到batch维度
    odd_mask_expanded = odd_mask.unsqueeze(0).expand(batch_size, -1)
    even_mask_expanded = even_mask.unsqueeze(0).expand(batch_size, -1)
    
    # 提取非填充的奇数位和偶数位数据
    odd_non_padding = time_data[odd_mask_expanded].reshape(batch_size, -1, time_dim)
    even_non_padding = time_data[even_mask_expanded].reshape(batch_size, -1, time_dim)
    
    # 计算每个batch中奇数位和偶数位的有效长度（时间数据的填充值通常为-1）
    # 检查所有5个维度是否都为-1来判断是否为填充
    padding_mask = (time_data == -1).all(dim=-1)
    odd_valid_lengths = torch.sum(~padding_mask & odd_mask_expanded, dim=1)
    even_valid_lengths = torch.sum(~padding_mask & even_mask_expanded, dim=1)
    
    # 初始化奇数位和偶数位数据，全部设为填充值-1
    odd_data = torch.full_like(time_data, -1)
    even_data = torch.full_like(time_data, -1)
    
    # 将连续的数据填充到结果中
    for i in range(batch_size):
        # 奇数位数据
        if odd_valid_lengths[i] > 0:
            odd_data[i, :odd_valid_lengths[i]] = odd_non_padding[i, :odd_valid_lengths[i]]
        
        # 偶数位数据
        if even_valid_lengths[i] > 0:
            even_data[i, :even_valid_lengths[i]] = even_non_padding[i, :even_valid_lengths[i]]
    
    return odd_data, even_data, org_time_data

# 设置日志
def setup_logging(log_dir):
    """设置日志记录"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('EvalRunner')



# 表征向量函数（get_seq_emb_from_traj_模型名_任务名）
def get_seq_emb_from_traj_TrajFuse_time(model, eval_data, logger, batch_size=64):
    """TrajFuse从轨迹数据中提取序列嵌入（fused_representation）"""
    device = next(model.parameters()).device
    model.eval()

    fused_representation_list=[]
    total_time_list=[]

    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_data, desc='Evaluating')):
            try:
                # 提取基本数据
                gps_data = batch['gps_data'].to(device) if batch['gps_data'] is not None else None
                cdr_data = batch['cdr_data'].to(device) if batch['cdr_data'] is not None else None
                road_seg_data = batch['road_gps_data'].to(device) if batch['road_gps_data'] is not None else None

                # 获取每条轨迹的总时间
                total_time = batch['total_time'].to(device) if 'total_time' in batch else None

                
                # 提取扩展数据字段
                kwargs = {}               
                # 提取扩展时间字段
                if 'gps_time_data' in batch and batch['gps_time_data'] is not None:
                    kwargs['gps_time_data'] = batch['gps_time_data'].to(device)
                if 'cdr_time_data' in batch and batch['cdr_time_data'] is not None:
                    kwargs['cdr_time_data'] = batch['cdr_time_data'].to(device)
                if 'road_gps_time_data' in batch and batch['road_gps_time_data'] is not None:
                    kwargs['road_time_data'] = batch['road_gps_time_data'].to(device)

                # 健壮性处理
                road_time_data = kwargs.get('road_time_data',None)
                gps_time_data = kwargs.get('gps_time_data',None)
                cdr_time_data = kwargs.get('cdr_time_data',None)

                gps_data = model._robust_preprocess_input_data(gps_data, "GPS")
                cdr_data = model._robust_preprocess_input_data(cdr_data, "CDR") 
                road_seg_data = model._robust_preprocess_input_data(road_seg_data, "ROAD_GPS")

                road_time_data = model._robust_preprocess_time_data(road_time_data)
                gps_time_data = model._robust_preprocess_time_data(gps_time_data)
                cdr_time_data = model._robust_preprocess_time_data(cdr_time_data)

                gps_data_padding = (gps_data[:, :, 0] != -1.0) & (gps_data[:, :, 1] != -1.0)# 获取GPS数据的掩码（填充值为-1.0）   
                cdr_data_padding = (cdr_data[:, :, 0] != -1.0) & (cdr_data[:, :, 1] != -1.0)# 获取CDR数据的掩码（填充值为-1.0）
                road_data_padding = (road_seg_data[:, :, 0] != -1.0) & (road_seg_data[:, :, 1] != -1.0)        # 获取路段数据的掩码（填充值为-1.0）
                
                


                
                # 1. 编码各种模态 
                encoded_embeddings = {}
                time_positions={}
                
                
                try:
                    # GPS序列编码
                    if gps_data is not None:
                        time_positions['gps']=gps_time_data
                        #屏蔽时间信息
                        gps_time_embedding=model.time_encoder(gps_time_data)  

                        gps_time_embedding = torch.zeros_like(gps_time_embedding)   

     
                        gps_embedding,_,gps_traj_rep = model.gps_encoder(gps_data,gps_time_embedding,~gps_data_padding,model.share_input_projection,model.share_output_projection)            
                        gps_embedding = model._safe_tensor_operation(gps_embedding, "GPS编码")

                        
                        encoded_embeddings['gps'] = gps_embedding

                    # CDR序列编码
                    if cdr_data is not None :
                        time_positions['cdr']=cdr_time_data

                        cdr_time_embedding=model.time_encoder(cdr_time_data)

                        cdr_time_embedding = torch.zeros_like(cdr_time_embedding)   


                        cdr_embedding,_,cdr_traj_rep = model.cdr_encoder(cdr_data,cdr_time_embedding,~cdr_data_padding,model.share_input_projection,model.share_output_projection)
                        cdr_embedding = model._safe_tensor_operation(cdr_embedding, "CDR编码")

                        encoded_embeddings['cdr'] = cdr_embedding   

                    # 路段序列编码
                    if road_seg_data is not None:
                        time_positions['road_seg']=road_time_data

                        road_time_embedding=model.time_encoder(road_time_data) 
                        road_time_embedding = torch.zeros_like(road_time_embedding)      

                        road_seg_embedding,_,road_seg_rep = model.road_segment_encoder(road_seg_data,road_time_embedding,~road_data_padding,model.share_input_projection,model.share_output_projection)
                        road_seg_embedding = model._safe_tensor_operation(road_seg_embedding, "GPS编码")


                        encoded_embeddings['road_seg'] = road_seg_embedding

                except Exception as e:
                    print(f"编码阶段发生错误1: {e}")
                    exit(1)            
                
                # 如果没有任何输入，返回默认结果
                if not encoded_embeddings:
                    print("没有有效的输入模态，模型终止")
                    exit(1)
                
                # 2. 多模态融合
                try:                
                    fusion_inputs = {}
                    fusion_padding_mask = {}
                    fusion_time_positions = {}
                    if 'gps' in encoded_embeddings:
                        fusion_inputs['gps'] = encoded_embeddings['gps']
                        fusion_padding_mask['gps'] = ~gps_data_padding
                        fusion_time_positions['gps']=time_positions['gps']
                    if 'cdr' in encoded_embeddings:
                        fusion_inputs['cdr'] = encoded_embeddings['cdr']
                        fusion_padding_mask['cdr'] = ~cdr_data_padding
                        fusion_time_positions['cdr']=time_positions['cdr']  
                    if 'road_seg' in encoded_embeddings:
                        fusion_inputs['road_seg'] = encoded_embeddings['road_seg']
                        fusion_padding_mask['road_seg'] = ~road_data_padding
                        fusion_time_positions['road_seg']=time_positions['road_seg']
                            

                    fused_representation,fused_representation_padding_mask = model.modal_fusion(fusion_inputs,None,fusion_padding_mask,fusion_time_positions)
                    fused_representation = model._safe_tensor_operation(fused_representation, "模态融合") 

                    # 对序列维度进行平均池化，得到每个样本的单一表示
                    if fused_representation.dim() == 3:  # (batch, seq_len, feature_dim)
                        # 使用平均池化将序列维度压缩为单一向量
                        fused_representation = torch.mean(fused_representation, dim=1)  
            
                    
                except Exception as e:
                    print(f"推理阶段发生错误2: {e}")
                    exit(1)
                
                #收集融合表示和总时间
                fused_representation_list.append(fused_representation)
                total_time_list.append(total_time)

            except Exception as e:
                logger.error(f"评估批次 {batch_idx} 发生错误: {e}")
                continue

    if fused_representation_list:
        fused_representation_all = torch.cat(fused_representation_list, dim=0)
        total_time_all = torch.cat(total_time_list, dim=0)
        print(f"提取的嵌入形状: {fused_representation_all.shape}, 总时间形状: {total_time_all.shape}")

    return fused_representation_all,total_time_all

def get_seq_emb_from_traj_TrajFuse_retrieval(model, eval_data, logger,disturbance=False, batch_size=64):
    """TrajFuse从轨迹数据中提取序列嵌入（fused_representation）
        输入：
        model: TrajFuse模型实例
        eval_data: 评估数据集（DataLoader）
        logger: 日志记录器
        disturbance: 是否对query添加扰动（默认False，同源查询时才加）
        batch_size: 批次大小（默认64）
        
    """
    device = next(model.parameters()).device
    model.eval()

    gps_embedding_list=[]
    cdr_embedding_list=[]
    road_seg_embedding_list=[]
    gps_query_embedding_list=[]
    gps_truth_embedding_list=[]



    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_data, desc='Evaluating')):
            try:
                # 提取基本数据
                gps_data = batch['gps_data'].to(device) if batch['gps_data'] is not None else None
                cdr_data = batch['cdr_data'].to(device) if batch['cdr_data'] is not None else None
                road_seg_data = batch['road_gps_data'].to(device) if batch['road_gps_data'] is not None else None
                target_seg_data = batch['target_seg_data'].to(device)
                road_network = batch['road_network'].to(device)

                road_time_data = batch['road_gps_time_data'].to(device) if 'road_gps_time_data' in batch else None
                gps_time_data = batch['gps_time_data'].to(device) if 'gps_time_data' in batch else None
                cdr_time_data = batch['cdr_time_data'].to(device) if 'cdr_time_data' in batch else None

                gps_truth_dara, gps_query_data, gps_data  = split_gps_data_by_odd_even(gps_data)
                gps_truth_time_data, gps_query_time_data, gps_time_data = split_time_data_by_odd_even(gps_time_data)


                road_time_data = model._robust_preprocess_time_data(road_time_data)
                gps_time_data = model._robust_preprocess_time_data(gps_time_data)
                cdr_time_data = model._robust_preprocess_time_data(cdr_time_data)
                gps_truth_time_data = model._robust_preprocess_time_data(gps_truth_time_data)
                gps_query_time_data = model._robust_preprocess_time_data(gps_query_time_data)

                
                gps_data = model._robust_preprocess_input_data(gps_data, "GPS")
                cdr_data = model._robust_preprocess_input_data(cdr_data, "CDR") 
                road_seg_data = model._robust_preprocess_input_data(road_seg_data,"ROAD_GPS")
                gps_truth_dara = model._robust_preprocess_input_data(gps_truth_dara, "GPS")
                gps_query_data = model._robust_preprocess_input_data(gps_query_data, "GPS")

                gps_data_padding = (gps_data[:, :, 0] != -1.0) & (gps_data[:, :, 1] != -1.0)# 获取GPS数据的掩码（填充值为-1.0）   
                cdr_data_padding = (cdr_data[:, :, 0] != -1.0) & (cdr_data[:, :, 1] != -1.0)# 获取CDR数据的掩码（填充值为-1.0）
                road_data_padding = (road_seg_data[:, :, 0] != -1.0) & (road_seg_data[:, :, 1] != -1.0)        # 获取路段数据的掩码（填充值为-1）
                gps_truth_dara_padding = (gps_truth_dara[:, :, 0] != -1.0) & (gps_truth_dara[:, :, 1] != -1.0)# 获取GPS数据的掩码（填充值为-1.0）   
                gps_query_data_padding = (gps_query_data[:, :, 0] != -1.0) & (gps_query_data[:, :, 1] != -1.0)# 获取GPS数据的掩码（填充值为-1.0）

                
                # 1. 编码各种模态 
                gps_time_embedding = model.time_encoder(gps_time_data) if gps_time_data is not None else None
                cdr_time_embedding = model.time_encoder(cdr_time_data) if cdr_time_data is not None else None
                road_time_embedding = model.time_encoder(road_time_data) if road_time_data is not None else None 
                gps_truth_time_embedding = model.time_encoder(gps_truth_time_data) if gps_truth_time_data is not None else None
                gps_query_time_embedding = model.time_encoder(gps_query_time_data) if gps_query_time_data is not None else None
                
                
                try:
                    # GPS序列编码
                    if gps_data is not None:                 
                        gps_embedding,_,gps_traj = model.gps_encoder(gps_data,gps_time_embedding,~gps_data_padding,model.share_input_projection,model.share_output_projection)            

                        # 对GPS序列进行扰动
                        if disturbance:
                            gps_query_embedding,_,gps_query_traj = model.gps_encoder(gps_query_data,gps_query_time_embedding,~gps_query_data_padding,model.share_input_projection,model.share_output_projection)
                            gps_truth_embedding,_,gps_truth_traj = model.gps_encoder(gps_truth_dara,gps_truth_time_embedding,~gps_truth_dara_padding,model.share_input_projection,model.share_output_projection)


                    # CDR序列编码
                    if cdr_data is not None:
                        cdr_embedding,_,cdr_traj = model.cdr_encoder(cdr_data,cdr_time_embedding,~cdr_data_padding,model.share_input_projection,model.share_output_projection)



                    # 路段序列编码
                    if road_seg_data is not None:
                        road_seg_embedding,_,road_traj = model.road_segment_encoder(road_seg_data,road_time_embedding,~road_data_padding,model.share_input_projection,model.share_output_projection)



                except Exception as e:
                    print(f"编码阶段发生错误: {e}")
                    exit(1)            
                
                           
                #收集表示
                gps_embedding_list.append(gps_traj)
                cdr_embedding_list.append(cdr_traj)
                road_seg_embedding_list.append(road_traj)

                if disturbance:
                    gps_query_embedding_list.append(gps_query_traj)
                    gps_truth_embedding_list.append(gps_truth_traj)




            except Exception as e:
                logger.error(f"评估批次 {batch_idx} 发生错误: {e}")
                continue

    gps_embedding_all = torch.cat(gps_embedding_list, dim=0)
    cdr_embedding_all = torch.cat(cdr_embedding_list, dim=0)
    road_seg_embedding_all = torch.cat(road_seg_embedding_list, dim=0)
    if disturbance:
        gps_query_embedding_all = torch.cat(gps_query_embedding_list, dim=0)
        gps_truth_embedding_all = torch.cat(gps_truth_embedding_list, dim=0)
    else:
        gps_query_embedding_all = None
        gps_truth_embedding_all = None
    


    return gps_embedding_all,cdr_embedding_all,road_seg_embedding_all,gps_query_embedding_all,gps_truth_embedding_all

class EvalDataLoader:
    """评估数据加载器"""
    def __init__(self, data_path, logger, batch_size=64, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.logger = logger
        self.num_workers = num_workers

        
        
        # 创建数据加载器
        _, self.eval_loader, self.eval_road_network, self.eval_normalization_stats = get_trifusion_loaders(
            data_path=self.data_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            route_min_len=5,
            route_max_len=256,
            gps_min_len=10,
            gps_max_len=256,
            cdr_min_len=2,
            cdr_max_len=256,
            num_samples=25000,  # 使用全部样本
            test_ratio=1.0,  # 全部作为测试数据
            modal_dropout_prob=0.0,  # 测试时不使用modal dropout
            apply_modal_dropout_train=False,
            apply_modal_dropout_test=False,
            seed=2023
        )
        self.logger.info(f'评估模式：评估数据加载成功: {len(self.eval_loader.dataset)} 评估样本')

    def get_eval_loader(self):
        return self.eval_loader, self.eval_road_network, self.eval_normalization_stats

class EvalRunner:
    """综合评估运行器"""
    
    def __init__(self, model_name, model_path, data_path, log_dir='./eval_logs'):
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.log_dir = log_dir
        self.logger = setup_logging(log_dir)
        self.device = torch.device(f'cuda:{dev_id}' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型和数据
        self.model = None
        self.eval_data_loader = None
        self.road_network = None
        self.eval_data = None
        self.eval_normalization_stats = None

        self._load_data()  
        self._load_model()
        
    def _load_model(self):
        if self.model_name == 'TrajFuse':
            self._get_TrajFuse_model()
        elif self.model_name == 'ConDTC':
            self._get_ConDTC_model()
            
        else:
            self.logger.error(f"模型加载失败，不支持的模型名称: {self.model_name}")

            raise ValueError(f"不支持的模型名称: {self.model_name}")
        
        self.logger.info(f"{self.model_name}模型加载完成")
    
    def _get_TrajFuse_model(self):
        """加载训练好的模型"""
        self.logger.info(f"加载模型: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 获取road_network_size（从checkpoint或默认值）
            if self.road_network is not None and hasattr(self.road_network, 'x'):
                road_network_size = checkpoint.get('road_network_size', self.road_network.x.size(0))
            else:
                road_network_size = checkpoint.get('road_network_size', 6450)  # 默认值
            
            # 重新构建模型
            self.model = TrajFuse(
                d_model=256,
                nhead=4,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=512,
                road_network_size=road_network_size,
                dropout=0.1,
                generation_mode='parallel',
                normalization_stats = self.eval_normalization_stats
            ).to(self.device)
            
            # 加载模型状态
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.logger.info("模型加载成功")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise

    def _load_data(self):
        """加载测试数据"""
        self.logger.info(f"加载数据: {self.data_path}")
        
        try:
            # 加载评估数据
            self.eval_data_loader = EvalDataLoader(
                data_path=self.data_path,
                logger=self.logger,
                batch_size=64,
                num_workers=0
            )
            self.eval_data, self.road_network, self.eval_normalization_stats = self.eval_data_loader.get_eval_loader()
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise

 
    def evaluate_travel_time_estimation(self, batch_size=64, folds=5):
        """评估旅行时间估计任务"""
        self.logger.info("=" * 50)
        self.logger.info(f"模型{self.model_name}开始旅行时间估计评估...")
        self.logger.info(f"使用{folds}折交叉验证，批次大小: {batch_size}")
        start_time = time.time()
        self.logger.info(f"评估开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        # 提取序列嵌入
        self.logger.info("提取序列嵌入...")
        if self.model_name == 'TrajFuse':
            seq_embedding,total_time = get_seq_emb_from_traj_TrajFuse_time(self.model, self.eval_data, self.logger, batch_size=64)
        else:
            self.logger.error(f"请补充实现模型{self.model_name}的旅行时间估计评估函数")
            exit(1)
            
        

        if seq_embedding is None or total_time is None:
            self.logger.error("序列嵌入提取失败或总时间数据缺失，旅行时间评测终止")
            exit(1)
            return None
        
        # 准备数据
        y = total_time.squeeze()  # 真实旅行时间标签
        x = seq_embedding  # 序列嵌入特征
        
        if len(x) != len(y):
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
            self.logger.warning(f"特征和标签长度不匹配，截断到 {min_len}")
        
        # 交叉验证评估
        split = x.shape[0] // folds
        self.logger.info(f"使用{folds}折交叉验证，每折{split}个样本，共{x.shape[0]}个样本")
        
        
        device_flag = True
        fold_preds = []
        fold_trues = []
        
        for i in range(folds):
            self.logger.info(f"处理第 {i+1} 折...")
            
            eval_idx = list(range(i * split, (i + 1) * split, 1))
            train_idx = list(set(list(range(x.shape[0]))) - set(eval_idx))

            x_train, x_eval = x[train_idx], x[eval_idx]
            y_train, y_eval = y[train_idx], y[eval_idx]
            
            fold_trues.append(y_eval)
            
            
            # 标准化标签
            y_train, mean, std = label_norm(y_train)
            
            # 初始化回归模型
            model = MLPReg(x.shape[1], 3, nn.ReLU()).cuda()

            if device_flag:
                print('MLPReg model device: ', next(model.parameters()).device)
                device_flag = False
            
            # 训练参数
            patience = 3
            epoch_threshold = 10
            epoch_num = 30
            best_epoch = 0
            best_mae = 1e9
            best_rmse = 1e9
            best_preds = None

            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            for epoch in range(1, epoch_num + 1):
                model.train()     
                # 小批次训练
                for batch_index in next_batch_index(x_train.shape[0], batch_size):
                    opt.zero_grad()
                    x_batch = x_train[batch_index].cuda()  # 移动到CUDA
                    y_batch = y_train[batch_index].cuda()  # 移动到CUDA
                    loss = nn.MSELoss()(model(x_batch), y_batch)
                    loss.backward()
                    opt.step()
                
                # 验证
                model.eval()
                y_preds = model(x_eval.cuda()).detach().cpu()  # x_eval也需要移动到CUDA
                y_preds = pred_unnorm(y_preds, mean, std)
                mae = mean_absolute_error(y_eval.cpu(), y_preds)  # 将y_eval移动到CPU
                rmse = mean_squared_error(y_eval.cpu(), y_preds) ** 0.5  # 将y_eval移动到CPU
                print(f'Fold {i+1}, Epoch: {epoch}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
                
                if epoch == epoch_num:
                    fold_preds.append(best_preds if best_preds is not None else y_preds)

                if mae < best_mae and rmse < best_rmse:
                    best_preds = y_preds
                    best_mae = mae
                    best_rmse = rmse
                    best_epoch = epoch
                    patience = 3
                else:
                    if epoch > epoch_threshold:
                        patience -= 1
                    if patience < 0:
                        fold_preds.append(best_preds)
                        break
        
        # 计算总体指标
        y_preds = torch.cat(fold_preds, dim=0)
        y_trues = torch.cat(fold_trues, dim=0)
        
        mae = mean_absolute_error(y_trues.cpu(), y_preds)  # 将y_trues移动到CPU
        rmse = mean_squared_error(y_trues.cpu(), y_preds) ** 0.5  # 将y_trues移动到CPU
        
        results = {
            'best_epoch': best_epoch,
            'mae': mae,
            'rmse': rmse
        }
        
        self.logger.info(f"旅行时间估计结果 - 最佳轮次: {best_epoch}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        end_time = time.time()
        cost_time = end_time - start_time
        self.logger.info(f"评估结束耗时: {cost_time:.4f} 秒")
        self.logger.info("=" * 50)
        
        return results
      
    def evaluate_similarity_retrieval(self, query_type='gps_query_gps'):
        """评估相似度检索任务"""
        self.logger.info("=" * 50)
        self.logger.info(f"模型{self.model_name}开始相似度检索评估 - 查询类型: {query_type}")

        if query_type not in ['gps_query_road','cdr_query_road','gps_query_cdr','gps_query_gps']:
            self.logger.error(f"不支持的查询类型: {query_type}")
            return None
        disturbance=False

        # 同源才需要扰动
        if query_type == 'gps_query_gps' or query_type == 'road_query_road':
            disturbance=True
        else:
            disturbance=False

        if self.model_name == 'TrajFuse':
            gps_embedding,cdr_embedding,road_seg_embedding,gps_query_embedding,gps_truth_embedding = get_seq_emb_from_traj_TrajFuse_retrieval(self.model, self.eval_data,self.logger,disturbance=disturbance,batch_size=64)
            nums = gps_embedding.shape[0]


        if query_type == 'gps_query_gps' :
            self.logger.info("开始同源GPS查询GPS评估...")
            # 特征提取
            query_embedding = gps_query_embedding[:nums//5] # 扰动
            truth_embedding = gps_truth_embedding[:nums//5]
            other_data_gps_embedding = gps_embedding[nums//5:]

            matches = np.where(np.all(query_embedding.cpu().numpy() == query_embedding.cpu().numpy()[0], axis=1))[0]
            self.logger.info(f"查询0在查询库中的出现次数: {len(matches)}")
            self.logger.info(f"匹配索引列表: {matches.tolist()}")

            D_embedding = torch.cat((truth_embedding, other_data_gps_embedding), dim=0)

            self.logger.info("特征提取完成。")

            # 转换为numpy数组并归一化
            query_features = F.normalize(query_embedding.cpu(), dim=1).numpy()  
            D_features = F.normalize(D_embedding.cpu(), dim=1).numpy()  
                
            self.logger.info(f"查询特征维度: {query_features.shape}")  
            self.logger.info(f"数据库特征维度: {D_features.shape}")  


            # 构建索引
            query_indices = np.arange(len(query_features))

            metrics=search_and_evaluate(query_features, D_features, query_indices)
            

            # 输出JSON格式结果
            self.logger.info(json.dumps({
                "query_type": query_type,
                "status": "success",
                "metrics": metrics,
                "feature_dim": query_features.shape[1],
                "index_type": "IVFFlat",
                "normalization": "L2"
            }, indent=2))

            self.logger.info("同源GPS查询GPS评估完成。")
        
        elif query_type == 'gps_query_road' :
            self.logger.info("开始跨源GPS查询Road评估...")
            # 特征提取
            query_embedding = gps_embedding[:nums//5] 
            truth_embedding = road_seg_embedding[:nums//5]
            other_data_gps_embedding = road_seg_embedding[nums//5:]

            matches = np.where(np.all(query_embedding.cpu().numpy() == query_embedding.cpu().numpy()[0], axis=1))[0]
            self.logger.info(f"查询0在查询库中的出现次数: {len(matches)}")
            self.logger.info(f"匹配索引列表: {matches.tolist()}")

            D_embedding = torch.cat((truth_embedding, other_data_gps_embedding), dim=0)


            self.logger.info("特征提取完成。")

            # 转换为numpy数组并归一化
            query_features = F.normalize(query_embedding.cpu(), dim=1).numpy()  
            D_features = F.normalize(D_embedding.cpu(), dim=1).numpy()  
            

                
            self.logger.info(f"查询特征维度: {query_features.shape}")  
            self.logger.info(f"数据库特征维度: {D_features.shape}")  


            # 构建索引
            query_indices = np.arange(len(query_features))


            metrics=search_and_evaluate(query_features, D_features, query_indices)


            # 输出JSON格式结果
            self.logger.info(json.dumps({
                "query_type": query_type,
                "status": "success",
                "metrics": metrics,
                "feature_dim": query_features.shape[1],
                "index_type": "IVFFlat",
                "normalization": "L2"
            }, indent=2))

            self.logger.info("跨源GPS查询Road评估完成。")

        elif query_type == 'gps_query_cdr' :
            self.logger.info("开始跨源GPS查询CDR评估...")
            # 特征提取
            query_embedding = gps_embedding[:nums//5] 
            truth_embedding = cdr_embedding[:nums//5]
            other_data_gps_embedding = cdr_embedding[nums//5:]

            matches = np.where(np.all(query_embedding.cpu().numpy() == query_embedding.cpu().numpy()[0], axis=1))[0]
            self.logger.info(f"查询0在查询库中的出现次数: {len(matches)}")
            self.logger.info(f"匹配索引列表: {matches.tolist()}")

            D_embedding = torch.cat((truth_embedding, other_data_gps_embedding), dim=0)


            self.logger.info("特征提取完成。")

            # 转换为numpy数组并归一化
            query_features = F.normalize(query_embedding.cpu(), dim=1).numpy()  
            D_features = F.normalize(D_embedding.cpu(), dim=1).numpy()  
            

                
            self.logger.info(f"查询特征维度: {query_features.shape}")  
            self.logger.info(f"数据库特征维度: {D_features.shape}")  


            # 构建索引
            query_indices = np.arange(len(query_features))


            metrics=search_and_evaluate(query_features, D_features, query_indices)


            # 输出JSON格式结果
            self.logger.info(json.dumps({
                "query_type": query_type,
                "status": "success",
                "metrics": metrics,
                "feature_dim": query_features.shape[1],
                "index_type": "IVFFlat",
                "normalization": "L2"
            }, indent=2))

            self.logger.info("跨源GPS查询CDR评估完成。")

        elif query_type == 'cdr_query_road' :
            self.logger.info("开始跨源CDR查询Road评估...")
            # 特征提取
            query_embedding = cdr_embedding[:nums//5] 
            truth_embedding = road_seg_embedding[:nums//5]
            other_data_gps_embedding = road_seg_embedding[nums//5:]

            matches = np.where(np.all(query_embedding.cpu().numpy() == query_embedding.cpu().numpy()[0], axis=1))[0]
            self.logger.info(f"查询0在查询库中的出现次数: {len(matches)}")
            self.logger.info(f"匹配索引列表: {matches.tolist()}")

            D_embedding = torch.cat((truth_embedding, other_data_gps_embedding), dim=0)


            self.logger.info("特征提取完成。")

            # 转换为numpy数组并归一化
            query_features = F.normalize(query_embedding.cpu(), dim=1).numpy()  
            D_features = F.normalize(D_embedding.cpu(), dim=1).numpy()  
            

                
            self.logger.info(f"查询特征维度: {query_features.shape}")  
            self.logger.info(f"数据库特征维度: {D_features.shape}")  


            # 构建索引
            query_indices = np.arange(len(query_features))


            metrics=search_and_evaluate(query_features, D_features, query_indices)


            # 输出JSON格式结果
            self.logger.info(json.dumps({
                "query_type": query_type,
                "status": "success",
                "metrics": metrics,
                "feature_dim": query_features.shape[1],
                "index_type": "IVFFlat",
                "normalization": "L2"
            }, indent=2))

            self.logger.info("跨源CDR查询Road评估完成。")

        else:
            self.logger.error(f"不支持的查询类型: {query_type}")
            self.logger.error(f"请补充该查询类型")
            return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='综合评估运行器')
    parser.add_argument('--model-name', type=str, default='TrajFuse', choices=['TrajFuse', 'ConDTC', 'GREEN', 'START', 'JGRM','TrajMamba','TrajCL','Word2Vec','Node2Vec'], help='模型名称')
    parser.add_argument('--model-path', type=str, default='/root/TrajFuse_代码/checkpoints/消融7.8/只留第二层对比学习/best_model.pt', help='模型文件路径')
    parser.add_argument('--data-path', type=str, default='/root/autodl-tmp/data/chengdu_eval_extension65_50_35_road2gps_gps150.pkl', help='测试数据路径')
    parser.add_argument('--log-dir', type=str, default='/root/TrajFuse_代码/eval_logs', help='评测日志目录')
    parser.add_argument('--task', type=str, default='all', 
                       choices=[ 'time', 'retrieval','all'],
                       help='评估任务类型')


    
    args = parser.parse_args()
    
    # 创建评估运行器
    runner = EvalRunner(args.model_name, args.model_path, args.data_path, args.log_dir)
    
    # 运行指定任务

    if args.task == 'time':
        results = runner.evaluate_travel_time_estimation()
    elif args.task == 'retrieval':
        results = runner.evaluate_similarity_retrieval('gps_query_gps')
        results = runner.evaluate_similarity_retrieval('gps_query_road')

    else:
        results = runner.evaluate_travel_time_estimation()
        results = runner.evaluate_similarity_retrieval('gps_query_gps')
        results = runner.evaluate_similarity_retrieval('gps_query_road')    
        results = runner.evaluate_similarity_retrieval('gps_query_cdr')
        results = runner.evaluate_similarity_retrieval('cdr_query_road')
        
    

if __name__ == '__main__':
    main()