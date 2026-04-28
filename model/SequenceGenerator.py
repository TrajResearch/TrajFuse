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


class SequenceGenerator(nn.Module):
    """序列生成器 - 支持并行生成所有token和自回归生成"""
    def __init__(self, d_model=256, nhead=4, num_layers=3, road_network_size=10000, dropout=0.1, generation_mode='parallel'):
        super(SequenceGenerator, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = road_network_size
        self.generation_mode = generation_mode

        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 路段嵌入和位置编码
        self.segment_embedding = nn.Embedding(road_network_size, d_model)
        nn.init.normal_(self.segment_embedding.weight, mean=0.0, std=0.02)
        self.position_encoding = PositionalEncoding(d_model)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, road_network_size)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.01)
        
        # 起始标记 - 使用确定性初始化
        self.start_token = nn.Parameter(torch.zeros(1, d_model))
        nn.init.normal_(self.start_token, mean=0.0, std=0.01)
        
        # 并行生成的可学习位置查询 - 使用确定性初始化
        self.parallel_queries = nn.Parameter(torch.zeros(500, d_model))
        nn.init.normal_(self.parallel_queries, mean=0.0, std=0.01)
    
    def set_generation_mode(self, mode):
        """设置生成模式
        
        Args:
            mode: 'parallel' 或 'autoregressive'
        """
        if mode not in ['parallel', 'autoregressive']:
            raise ValueError(f"不支持的生成模式: {mode}，请选择 'parallel' 或 'autoregressive'")
        self.generation_mode = mode
        
    def forward(self, fused_representation,fused_representation_padding_mask=None, target=None, max_length=None, use_parallel=None):
        """序列生成 - 根据generation_mode统一控制训练和推理阶段的生成方式"""
        batch_size = fused_representation.size(0)
        device = fused_representation.device
        
        if max_length is None:
            if target is not None:
                max_length = target.size(1)
            else:
                max_length = 50
        
        # 根据generation_mode决定生成方式，统一训练和推理阶段
        if self.generation_mode == 'parallel':
            # 并行生成模式：训练和推理都使用并行解码
            if self.training and target is not None:
                return self._parallel_teacher_forcing_decode(fused_representation, fused_representation_padding_mask, target)
            else:
                return self._parallel_decode(fused_representation, fused_representation_padding_mask, max_length)
        elif self.generation_mode == 'autoregressive':
            # 自回归生成模式：训练和推理都使用自回归解码
            if self.training and target is not None:
                return self._teacher_forcing_decode(fused_representation, target)
            else:
                return self._autoregressive_decode(fused_representation, max_length)
        else:
            raise ValueError(f"不支持的生成模式: {self.generation_mode}，请选择 'parallel' 或 'autoregressive'")
    
    def _teacher_forcing_decode(self, memory, target):
        """训练时的teacher forcing"""
        batch_size, target_len = target.shape
        device = memory.device
        
        try:
            # 准备解码器输入
            target_clamped = torch.clamp(target, 0, self.vocab_size-1)
            start_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            decoder_input_ids = torch.cat([start_tokens, target_clamped[:, :-1]], dim=1)
            
            # 嵌入
            decoder_input = self.segment_embedding(decoder_input_ids)
            decoder_input[:, 0] = self.start_token.expand(batch_size, -1)
            decoder_input = self.position_encoding(decoder_input)
            # 因果掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_len).to(device)
            
            # 解码
            decoder_output = self.transformer_decoder(
                decoder_input, memory, tgt_mask=tgt_mask
            )
            
            # 输出投影
            logits = self.output_projection(decoder_output)
            
            # 检查输出
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("警告：解码器输出异常")
                logits = torch.zeros_like(logits)
            
        except Exception as e:
            print(f"Teacher forcing解码错误: {e}")
            logits = torch.zeros(batch_size, target_len, self.vocab_size, device=device)
        
        generation_info = {'generation_mode': 'teacher_forcing'}
        return logits, generation_info
    
    def _parallel_teacher_forcing_decode(self, memory, memory_key_padding_mask, target):
        """并行teacher forcing解码 - 训练时也使用并行生成方式"""
        batch_size, target_len = target.shape
        device = memory.device

        memory_key_padding_mask=memory_key_padding_mask
        try:
            # 使用可学习的位置查询作为解码器输入，而不是目标嵌入
            decoder_queries = self.parallel_queries[:target_len].unsqueeze(0).expand(batch_size, -1, -1)
            
            # 添加位置编码
            decoder_input = self.position_encoding(decoder_queries)
            
            # 并行解码 - 不使用因果掩码，让所有位置同时生成
            decoder_output = self.transformer_decoder(decoder_input, memory,memory_key_padding_mask=memory_key_padding_mask)
            
            # 输出投影 - 一次性生成所有token的logits
            logits = self.output_projection(decoder_output)


            
            # 检查输出
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("警告：并行teacher forcing解码输出异常")
                logits = torch.zeros_like(logits)
            
        except Exception as e:
            print(f"并行teacher forcing解码错误: {e}")
            logits = torch.zeros(batch_size, target_len, self.vocab_size, device=device)
        
        generation_info = {'generation_mode': 'parallel_teacher_forcing'}
        return logits, generation_info
    
    def _parallel_decode(self, memory, memory_key_padding_mask, max_length):
        """并行解码 - 一次性生成所有token"""
        batch_size = memory.size(0)
        device = memory.device
        memory_key_padding_mask = memory_key_padding_mask
        
        try:
            # 使用可学习的位置查询作为解码器输入
            # 这些查询代表要生成的每个位置

            decoder_queries = self.parallel_queries[:max_length].unsqueeze(0).expand(batch_size, -1, -1)

            # 添加位置编码
            decoder_input = self.position_encoding(decoder_queries)

            # 并行解码 - 不使用因果掩码，让所有位置同时生成
            decoder_output= self.transformer_decoder(decoder_input, memory, memory_key_padding_mask=memory_key_padding_mask)

            # 输出投影 - 一次性生成所有token的logits
            logits = self.output_projection(decoder_output)

            
            # 检查输出
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("警告：并行解码输出异常")
                logits = torch.zeros_like(logits)
            
        except Exception as e:
            print(f"并行解码错误: {e}")
            logits = torch.zeros(batch_size, max_length, self.vocab_size, device=device)
        
        generation_info = {'generation_mode': 'parallel'}
        return logits, generation_info
    


    def _autoregressive_decode(self, memory, max_length):
        """自回归解码 - 保留原有逐步生成方式（可选）"""
        batch_size = memory.size(0)
        device = memory.device
        
        try:
            # 初始化
            current_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            current_input = self.start_token.expand(batch_size, 1, -1)
            
            generated_logits = []
            
            for step in range(max_length):
                # 位置编码
                positioned_input = self.position_encoding(current_input)
                
                # 解码
                decoder_output = self.transformer_decoder(positioned_input, memory)
                
                # 预测下一个token
                step_logits = self.output_projection(decoder_output[:, -1:])
                generated_logits.append(step_logits)
                
                # 准备下一步
                if step < max_length - 1:
                    next_token_ids = torch.argmax(step_logits, dim=-1)
                    next_token_ids = torch.clamp(next_token_ids, 0, self.vocab_size-1)
                    current_ids = torch.cat([current_ids, next_token_ids], dim=1)
                    next_token_embed = self.segment_embedding(next_token_ids)
                    current_input = torch.cat([current_input, next_token_embed], dim=1)
            
            # 合并logits
            all_logits = torch.cat(generated_logits, dim=1)
            
        except Exception as e:
            print(f"自回归解码错误: {e}")
            all_logits = torch.zeros(batch_size, max_length, self.vocab_size, device=device)
        
        generation_info = {'generation_mode': 'autoregressive'}
        return all_logits, generation_info
