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



class TimeEncoding(nn.Module):
    """时间位置编码器，处理月份、周、小时、分钟数据，使用连续时间编码方法"""
    def __init__(self, d_model):
        super(TimeEncoding, self).__init__()
        self.d_model = d_model
        
        # 为每个时间维度创建独立的编码参数
        # 月份编码参数 (0-11)
        self.month_omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, d_model))).float(), requires_grad=True)
        self.month_bias = nn.Parameter(torch.zeros(d_model).float(), requires_grad=True)
        
        # 周编码参数 (0-6)
        self.week_omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, d_model))).float(), requires_grad=True)
        self.week_bias = nn.Parameter(torch.zeros(d_model).float(), requires_grad=True)
        
        # 小时编码参数 (0-23)
        self.hour_omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, d_model))).float(), requires_grad=True)
        self.hour_bias = nn.Parameter(torch.zeros(d_model).float(), requires_grad=True)
        
        # 分钟编码参数 (0-59)
        self.minute_omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, d_model))).float(), requires_grad=True)
        self.minute_bias = nn.Parameter(torch.zeros(d_model).float(), requires_grad=True)
        
        # 时间维度融合权重
        self.month_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        self.week_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        self.hour_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        self.minute_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        
        self.div_term = math.sqrt(1. / d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, time_data):
        """
        处理四维时间数据
        
        Args:
            time_data: 四维时间数据，形状为 [batch_size, seq_len, 4]
                      维度顺序：[月份(0-11), 周(0-6), 小时(0-23), 分钟(0-59)]
                      其中-1为填充位
        
        Returns:
            time_embedding: 时间编码后的特征，形状为 [batch_size, seq_len, embed_size]
        """

        if time_data is None:
            print("警告：time_data参数缺失，返回零张量")
            return None

        time_data = time_data  # (batch, seq_len, 5)
        batch_size, seq_len, _ = time_data.shape

        # 分离四个时间维度
        month_data = time_data[:, :, 0]  # 月份 [batch_size, seq_len]
        week_data = time_data[:, :, 1]   # 周 [batch_size, seq_len]
        hour_data = time_data[:, :, 2]   # 小时 [batch_size, seq_len]
        minute_data = time_data[:, :, 3] # 分钟 [batch_size, seq_len]

        # 处理填充值：将填充值(-1)映射为13 7 24 60，确保数值有效
        month_data = torch.where(month_data == -1, 13, month_data).float()
        week_data = torch.where(week_data == -1, 7, week_data).float()
        hour_data = torch.where(hour_data == -1, 24, hour_data).float()
        minute_data = torch.where(minute_data == -1, 60, minute_data).float()

        # 钳位数值确保在合理范围内
        month_data = torch.clamp(month_data, 1, 12)
        week_data = torch.clamp(week_data, 0, 7)
        hour_data = torch.clamp(hour_data, 0, 24)
        minute_data = torch.clamp(minute_data, 0, 60)

        try:
            # 为每个时间维度计算交替时间编码（sin和cos交替）
            # 月份编码
            month_phase = month_data.unsqueeze(-1) * self.month_omega.reshape(1, 1, -1) + self.month_bias.reshape(1, 1, -1)
            month_encode = torch.empty_like(month_phase)
            month_encode[:, :, 0::2] = torch.sin(month_phase[:, :, 0::2])  # 偶数位置使用sin
            month_encode[:, :, 1::2] = torch.cos(month_phase[:, :, 1::2])  # 奇数位置使用cos
            
            # 周编码
            week_phase = week_data.unsqueeze(-1) * self.week_omega.reshape(1, 1, -1) + self.week_bias.reshape(1, 1, -1)
            week_encode = torch.empty_like(week_phase)
            week_encode[:, :, 0::2] = torch.sin(week_phase[:, :, 0::2])
            week_encode[:, :, 1::2] = torch.cos(week_phase[:, :, 1::2])
            
            # 小时编码
            hour_phase = hour_data.unsqueeze(-1) * self.hour_omega.reshape(1, 1, -1) + self.hour_bias.reshape(1, 1, -1)
            hour_encode = torch.empty_like(hour_phase)
            hour_encode[:, :, 0::2] = torch.sin(hour_phase[:, :, 0::2])
            hour_encode[:, :, 1::2] = torch.cos(hour_phase[:, :, 1::2])
            
            # 分钟编码
            minute_phase = minute_data.unsqueeze(-1) * self.minute_omega.reshape(1, 1, -1) + self.minute_bias.reshape(1, 1, -1)
            minute_encode = torch.empty_like(minute_phase)
            minute_encode[:, :, 0::2] = torch.sin(minute_phase[:, :, 0::2])
            minute_encode[:, :, 1::2] = torch.cos(minute_phase[:, :, 1::2])

            # 使用可学习的权重融合四个时间维度的编码
            weights = torch.softmax(torch.stack([
                self.month_weight, self.week_weight, self.hour_weight, self.minute_weight
            ]), dim=0)
            
            time_embedding = (
                weights[0] * month_encode + 
                weights[1] * week_encode + 
                weights[2] * hour_encode + 
                weights[3] * minute_encode
            )
            
            # 应用缩放因子
            time_embedding = self.div_term * time_embedding
            
        except Exception as e:
            print(f"时间编码计算错误: {e}")
            time_embedding = torch.zeros(batch_size, seq_len, self.d_model, device=self.device)
        
        # 归一化处理
        time_embedding = self.norm(time_embedding)
        
        return time_embedding
    
    @property
    def device(self):
        return next(self.parameters()).device

