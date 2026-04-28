from datetime import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch

from model.PositionalEncoding import PositionalEncoding
from model.AttentionPooler import AttentionPooler

class SequentialEncoder(nn.Module):
    """修复版序列编码器 - 大幅简化"""
    def __init__(self, input_dim, d_model=256, nhead=4, num_layers=3, dropout=0.1, 
                 fourier_mapping_size=256, fourier_scale=1, use_fourier=True):
        super(SequentialEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.eps = 1e-8
        self.use_fourier = use_fourier
        self.fourier_mapping_size = fourier_mapping_size
        self.fourier_scale = fourier_scale
        
        # 傅里叶变换参数
        if self.use_fourier:
            # 创建高斯傅里叶特征映射矩阵
            self.register_buffer('B_gauss', torch.randn(fourier_mapping_size, input_dim) * fourier_scale)
            # 傅里叶变换后的维度
            self.fourier_output_dim = 2 * fourier_mapping_size  


        
        # 位置编码
        self.position_encoding = PositionalEncoding(d_model)
        
        # 简化的Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2,  # 减小FFN
            dropout=dropout,
            activation='gelu',
            batch_first=True, 
            norm_first=True  # Pre-LN更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attn_pool = AttentionPooler(d_model)

        self.output_norm = nn.LayerNorm(d_model)

        
        # 应用权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """保守的权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)  
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    def _apply_fourier_transform(self, x):
        """应用傅里叶变换到输入坐标"""
        if not self.use_fourier:
            return x
            
        # 确保输入是2D坐标
        if x.shape[-1] != 2:
            print(f"警告：傅里叶变换期望2D坐标，但输入维度为{x.shape[-1]}，跳过傅里叶变换")
            return x
            
        batch_size, seq_len, _ = x.shape
        
        # 将输入坐标映射到高维傅里叶空间
        # x: [batch_size, seq_len, 2]
        # B_gauss: [fourier_mapping_size, 2]
        # 计算 B_gauss * x^T
        x_projected = torch.matmul(x, self.B_gauss.t())  # [batch_size, seq_len, fourier_mapping_size]
        
        # 应用sin和cos变换
        sin_component = torch.sin(2 * math.pi * x_projected)
        cos_component = torch.cos(2 * math.pi * x_projected)
        
        # 拼接sin和cos分量
        fourier_features = torch.cat([sin_component, cos_component], dim=-1)  # [batch_size, seq_len, 2*fourier_mapping_size]
        
        return fourier_features

    def forward(self, x, time_data = None, src_key_padding_mask=None,share_input_projection=None,share_output_projection=None):
        # 输入检查
        if x is None or x.numel() == 0:
            batch_size = 1 if x is None else x.size(0)
            return torch.zeros(batch_size, 1, self.d_model, device=self.device)
        
        # 数值安全性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 限制输入范围
        x = torch.clamp(x, min=-10.0, max=10.0)

        # 应用傅里叶变换（如果启用）
        if self.use_fourier:
            try:
                x = self._apply_fourier_transform(x)
                # 检查傅里叶变换结果
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print("警告：傅里叶变换输出异常，跳过傅里叶变换")
                    # 如果傅里叶变换失败，使用原始输入但需要调整维度
                    if x.shape[-1] != self.input_dim:
                        x = torch.clamp(x, min=-10.0, max=10.0)  # 重新获取原始输入
            except Exception as e:
                print(f"傅里叶变换错误: {e}")
                # 出错时使用原始输入
                x = torch.clamp(x, min=-10.0, max=10.0)
        
        # 投影
        try:
            x = share_input_projection(x)
            # 检查投影后结果
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("警告：投影层输出异常，重新初始化")
                batch_size, seq_len = x.shape[:2]
                x = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        except Exception as e:
            print(f"投影层错误: {e}")
            batch_size, seq_len = x.shape[:2]
            x = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        # 时间编码
        if time_data is not None :
            try:
                x = x +  time_data
            except Exception as e:
                print(f"时间编码错误: {e}")
                x = x
        else:
            x = x

        # 位置编码
        x = self.position_encoding(x)
        
        # Transformer编码
        try:
            encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
            
            # 检查编码结果
            if torch.isnan(encoded).any() or torch.isinf(encoded).any():
                print("警告：Transformer编码异常，使用输入")
                encoded = x
                
        except Exception as e:
            print(f"Transformer编码错误: {e}")
            encoded = x
        
        # 输出标准化
        output_norm = self.output_norm(encoded)
        
        # 最终输出投影
        output_projection = share_output_projection(output_norm)
        traj_rep = self.attn_pool(output_projection, mask=src_key_padding_mask)
        

        # 最终检查
        if torch.isnan(output_projection).any() or torch.isinf(output_projection).any():
            print("警告：编码器最终输出异常")
            batch_size, seq_len = output_projection.shape[:2]
            output_projection = torch.zeros(batch_size, seq_len, self.d_model, device=output_projection.device)
        
        return output_projection,output_norm,traj_rep  #过投影，没过投影,轨迹表示
     
    @property
    def device(self):
        return next(self.parameters()).device

