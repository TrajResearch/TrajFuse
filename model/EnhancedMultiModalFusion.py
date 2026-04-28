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



class EnhancedMultiModalFusion(nn.Module):
    """增强的多模态融合模块"""

    def __init__(self, d_model=256, nhead=4, dropout=0.1,num_layers=3):
        super(EnhancedMultiModalFusion, self).__init__()

        self.d_model = d_model


        # transformer 融合
        # 简化的Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2,
            dropout=dropout,
            activation='gelu',
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.modal_dropout_prob=0.1



        self.modalities = ['gps', 'cdr', 'road_seg']
        self._modality_to_idx = {name: i for i, name in enumerate(self.modalities)}
        self.modality_emb = nn.Embedding(len(self.modalities), d_model)
        nn.init.xavier_uniform_(self.modality_emb.weight, gain=0.01)


    def _get_time_position(self, time_data):
        """
        获取单个模态的时间位置编码
        
        Args:
            time_data: 四维时间数据，形状为 [batch_size, seq_len, 4]
                      维度顺序：[月份(1-12), 周(0-6), 小时(0-23), 分钟(0-59)]
                      其中-1均为填充位
        
        Returns:
            time_positions: 时间位置编码，形状为 [batch_size, seq_len]
        """
        if time_data is None:
            return None
        batch_size, seq_len, _ = time_data.shape

        # 分离四个时间维度
        #month_data = time_data[:, :, 0]  # 月份 [batch_size, seq_len]
        #week_data = time_data[:, :, 1]   # 周 [batch_size, seq_len]
        hour_data = time_data[:, :, 2]   # 小时 [batch_size, seq_len]
        minute_data = time_data[:, :, 3] # 分钟 [batch_size, seq_len]

        # 创建掩码，检测填充位（-1）
        # 如果小时或分钟为-1，则该位置为填充位
        padding_mask = (hour_data == -1) | (minute_data == -1)
        
        # 计算时间位置编码0-1439
        time_positions = (hour_data * 60 + minute_data)  # [batch_size, seq_len]
        
        # 将填充位的时间位置编码恢复为-1
        time_positions[padding_mask] = -1

        return time_positions

    def _get_merge_position_encoding(self, gps, cdr, road_seg, pad_value=-1):
        """
        将三个模态的时间位置序列按时间顺序融合，固定长度为三模态长度之和，相同时间步全部保留
        
        Args:
            gps (Tensor or None): [batch_size, seq_len_gps]
            cdr (Tensor or None): [batch_size, seq_len_cdr]
            road_seg (Tensor or None): [batch_size, seq_len_road]
            pad_value (int): 填充值，默认 -1

        Returns:
            merged_tensor (Tensor): [batch_size, seq_len_total, 4]，其中：
                - dim 0: 时间值
                - dim 1: GPS 原始索引（非 GPS 来源则为 pad_value）
                - dim 2: Road 原始索引（非 Road 来源则为 pad_value）
                - dim 3: CDR 原始索引（非 CDR 来源则为 pad_value）
        """
        # 至少一个模态非 None
        tensors = [t for t in (gps, cdr, road_seg) if t is not None]
        if not tensors:
            raise ValueError("At least one of gps, cdr, or road_seg must be provided.")
        
        batch_size = tensors[0].size(0)
        device = tensors[0].device

        # 获取各模态长度
        len_gps = gps.size(1) if gps is not None else 0
        len_road = road_seg.size(1) if road_seg is not None else 0
        len_cdr = cdr.size(1) if cdr is not None else 0
        total_len = len_gps + len_road + len_cdr

        # 提前返回空情况
        if total_len == 0:
            return torch.empty(batch_size, 0, 4, dtype=torch.long, device=device).fill_(pad_value)

        # 初始化输出张量（全 pad_value）
        merged = torch.full((batch_size, total_len, 4), pad_value, dtype=torch.long, device=device)

        # 主循环：每个样本独立处理（因有效长度不一，难以完全向量化）
        for b in range(batch_size):
            offset = 0  # 当前写入位置

            # --- GPS ---
            if gps is not None:
                mask = gps[b] != pad_value
                valid_times = gps[b][mask]
                valid_idxs = torch.where(mask)[0]
                n = valid_times.numel()
                if n > 0:
                    merged[b, offset:offset+n, 0] = valid_times
                    merged[b, offset:offset+n, 1] = valid_idxs
                    offset += n

            # --- Road ---
            if road_seg is not None:
                mask = road_seg[b] != pad_value
                valid_times = road_seg[b][mask]
                valid_idxs = torch.where(mask)[0]
                n = valid_times.numel()
                if n > 0:
                    merged[b, offset:offset+n, 0] = valid_times
                    merged[b, offset:offset+n, 2] = valid_idxs
                    offset += n

            # --- CDR ---
            if cdr is not None:
                mask = cdr[b] != pad_value
                valid_times = cdr[b][mask]
                valid_idxs = torch.where(mask)[0]
                n = valid_times.numel()
                if n > 0:
                    merged[b, offset:offset+n, 0] = valid_times
                    merged[b, offset:offset+n, 3] = valid_idxs
                    offset += n

            # --- 按时间排序（稳定排序保持模态插入顺序）---
            if offset > 1:
                times = merged[b, :offset, 0]
                # stable=True 确保相同时间下保持 GPS → Road → CDR 顺序
                sorted_idx = torch.argsort(times, stable=True)
                merged[b, :offset] = merged[b, :offset][sorted_idx]

        return merged

    def _get_fusion_embedding(self, modality_space_embeddings, time_positions):
        """
        根据融合后的时间位置编码，从各模态嵌入中提取对应向量进行融合。
        
        前提：每个有效时间步仅属于一个模态（无多模态重叠）。
        
        Args:
            modality_space_embeddings (dict): 包含 'gps', 'road_seg', 'cdr' 的嵌入字典，
                                            每个为 [batch_size, seq_len_modality, d_model] 或 None
            time_positions (Tensor): [batch_size, seq_len_total, 4]
                - [:,:,0]: 时间值
                - [:,:,1]: GPS 原始索引（非 GPS 为 -1）
                - [:,:,2]: Road 原始索引（非 Road 为 -1）
                - [:,:,3]: CDR 原始索引（非 CDR 为 -1）

        Returns:
            fused_embedding (Tensor): [batch_size, seq_len_total, d_model]
            time_position (Tensor): [batch_size, seq_len_total] —— 时间值
        """
        time_position = time_positions[:, :, 0]  # [B, L]
        gps_indices = time_positions[:, :, 1]     # [B, L]
        road_indices = time_positions[:, :, 2]    # [B, L]
        cdr_indices = time_positions[:, :, 3]     # [B, L]

        batch_size, seq_len = time_position.shape
        device = time_position.device

        d_model = next(iter(modality_space_embeddings.values())).size(-1)
        data_type = next(iter(modality_space_embeddings.values())).dtype

        # 初始化融合嵌入（用 0 填充无效位置，符合常规做法）
        fused_embedding = torch.zeros(batch_size, seq_len, d_model, dtype=data_type, device=device)

        # 有效位置掩码（非 pad）
        valid_mask = (time_position != -1)  # [B, L]

        # --- GPS ---
        if 'gps' in modality_space_embeddings and modality_space_embeddings['gps'] is not None:
            gps_emb = modality_space_embeddings['gps']  # [B, L_gps, D]
            # 找出属于 GPS 的有效位置（索引合法且在范围内）
            gps_mask = (gps_indices != -1) & (gps_indices < gps_emb.size(1)) & valid_mask  # [B, L]
            if gps_mask.any():
                batch_idx, seq_idx = torch.where(gps_mask)
                fused_embedding[batch_idx, seq_idx] = gps_emb[batch_idx, gps_indices[batch_idx, seq_idx]]

        # --- Road ---
        if 'road_seg' in modality_space_embeddings and modality_space_embeddings['road_seg'] is not None:
            road_emb = modality_space_embeddings['road_seg']  # [B, L_road, D]
            road_mask = (road_indices != -1) & (road_indices < road_emb.size(1)) & valid_mask
            if road_mask.any():
                batch_idx, seq_idx = torch.where(road_mask)
                fused_embedding[batch_idx, seq_idx] = road_emb[batch_idx, road_indices[batch_idx, seq_idx]]

        # --- CDR ---
        if 'cdr' in modality_space_embeddings and modality_space_embeddings['cdr'] is not None:
            cdr_emb = modality_space_embeddings['cdr']  # [B, L_cdr, D]
            cdr_mask = (cdr_indices != -1) & (cdr_indices < cdr_emb.size(1)) & valid_mask
            if cdr_mask.any():
                batch_idx, seq_idx = torch.where(cdr_mask)
                fused_embedding[batch_idx, seq_idx] = cdr_emb[batch_idx, cdr_indices[batch_idx, seq_idx]]

        return fused_embedding, time_position

    def forward(self, modality_embeddings, modality_time_embedding=None, modality_padding_masks=None, modality_time_positions=None):
        """
        融合多个模态的embedding
        
        Args:
            modality_embeddings: dict of embeddings {'modality_name': embedding_tensor}
            
        Returns:
            fused_embedding: 融合后的embedding
            padding_mask: 融合后的padding mask
        """
        if not modality_embeddings:
            raise ValueError("No modality embeddings provided")

        # 收集所有有效的embedding
        valid_embeddings = []
        valid_padding_masks = []
        modality_names = []

        for name, emb in modality_embeddings.items():
            if emb is not None and emb.numel() > 0:

                # 在每个模态的embedding上加上模态标识向量
                idx = self._modality_to_idx.get(name, 0)
                mod_vec = self.modality_emb(torch.tensor(idx, device=emb.device)).unsqueeze(0).unsqueeze(1)
                mod_vec = mod_vec.expand(emb.size(0), emb.size(1), -1)
                emb = emb + mod_vec
                modality_embeddings[name] = emb

                valid_embeddings.append(emb)
                valid_padding_masks.append(
                    modality_padding_masks[name] if modality_padding_masks and name in modality_padding_masks else None
                )
                modality_names.append(name)

        if not valid_embeddings:
            raise ValueError("No valid embeddings found")




        fused_embedding = None # 融合后的空间embedding
        padding_mask = None # 融合后的padding mask
        time_position = None # 融合后的时间位置编码
        fused_time_data = None # 融合后的时间数据
        fused_time_embedding = None # 融合后的时间embedding



        try:
            if modality_time_positions is not None:
                # 获取融合后的时间位置编码
                merged_time_positions = self._get_merge_position_encoding(
                    self._get_time_position(modality_time_positions.get('gps', None)),
                    self._get_time_position(modality_time_positions.get('cdr', None)),
                    self._get_time_position(modality_time_positions.get('road_seg', None)),
                    pad_value=-1
                )

                # 获取融合后的embedding
                fused_embedding, time_position = self._get_fusion_embedding(
                    modality_embeddings,
                    merged_time_positions
                )

                padding_mask = (time_position == -1)  # 填充位置为True，其他为False


            else:
                # 连接所有embedding
                fused_embedding = torch.cat(valid_embeddings, dim=1)

                padding_mask = torch.cat(valid_padding_masks, dim=1) if all(mask is not None for mask in valid_padding_masks) else None

                time_position = None



            # Transformer编码
            try:

                fused_features = self.transformer(fused_embedding, src_key_padding_mask=padding_mask)

                # 检查编码结果
                if torch.isnan(fused_features).any() or torch.isinf(fused_features).any():
                    print("警告：Transformer编码异常，使用自注意力")
                    exit(1)

                    
            except Exception as e:
                print(f"Transformer编码错误: {e}")
                exit(1)

            # 检查融合结果
            if torch.isnan(fused_features).any() or torch.isinf(fused_features).any():
                print("自注意力融合结果包含NaN或Inf，使用未融合特征代替")
                fused_features = fused_embedding

            # 输出投影
            output = fused_features

            if torch.isnan(output).any() or torch.isinf(output).any():
                output = fused_features

            return output, padding_mask 

        except Exception as e:
            print(f"多模态融合错误: {e}")
            exit(1)
    
