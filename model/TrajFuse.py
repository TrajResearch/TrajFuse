from datetime import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
random.seed(2023)
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch

from model.SequentialEncoder import SequentialEncoder
from model.SequenceGenerator import SequenceGenerator
from model.EnhancedMultiModalFusion import EnhancedMultiModalFusion
from model.TimeEncoding import TimeEncoding



class TrajFuse(nn.Module):

    def __init__(
        self, 
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        road_network_size=None,
        dropout=0.1,
        max_decode_len=500,
        main_loss_weight=1.0,  # 主任务损失权重
        contrastive_loss_weight=0.5,  #对比学习损失权重
        contrastive_temperature=0.1,  #对比学习温度参数
        sequence_reg_weight=0.1,  # 序列正则化损失权重
        normalization_stats=None,  # 标准化统计信息
        generation_mode='parallel',  # 生成模式: 'parallel' 或 'autoregressive'
        deterministic_seed=42,  # 确定性随机种子
        mlm_loss_weight=1.0,  # MLM任务损失权重
        mlm_mask_prob=0.15,  # MLM掩码概率
        mlm_replace_prob=0.8,  # MLM替换概率
        mlm_random_prob=0.1,  # MLM随机替换概率
    ):
        super(TrajFuse, self).__init__()
        self.d_model = d_model
        self.road_network_size = road_network_size or 10000
        self.max_decode_len = max_decode_len
        self.main_loss_weight = main_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight#对比学习损失权重
        self.contrastive_temperature = contrastive_temperature#对比学习温度参数
        self.sequence_reg_weight = sequence_reg_weight
        self.generation_mode = generation_mode
        
        # MLM任务参数
        self.mlm_loss_weight = mlm_loss_weight
        self.mlm_mask_prob = mlm_mask_prob
        self.mlm_replace_prob = mlm_replace_prob
        self.mlm_random_prob = mlm_random_prob
        
        # 创建确定性随机数生成器
        self.deterministic_generator = torch.Generator(device='cuda:0')
        self.deterministic_generator.manual_seed(deterministic_seed)
        
        # 保存标准化参数
        if normalization_stats is not None:
            self.gps_mean_lon = normalization_stats['gps_mean_lon']
            self.gps_mean_lat = normalization_stats['gps_mean_lat']
            self.gps_std_lon = normalization_stats['gps_std_lon']
            self.gps_std_lat = normalization_stats['gps_std_lat']
            self.cdr_mean_lon = normalization_stats['cdr_mean_lon']
            self.cdr_mean_lat = normalization_stats['cdr_mean_lat']
            self.cdr_std_lon = normalization_stats['cdr_std_lon']
            self.cdr_std_lat = normalization_stats['cdr_std_lat']
            self.road_gps_mean_lon = normalization_stats['road_gps_mean_lon']
            self.road_gps_mean_lat = normalization_stats['road_gps_mean_lat']
            self.road_gps_std_lon = normalization_stats['road_gps_std_lon']
            self.road_gps_std_lat = normalization_stats['road_gps_std_lat']
            
            # 添加0-1标准化参数
            self.gps_min_lon = normalization_stats['gps_min_lon']
            self.gps_max_lon = normalization_stats['gps_max_lon']
            self.gps_min_lat = normalization_stats['gps_min_lat']
            self.gps_max_lat = normalization_stats['gps_max_lat']
            self.cdr_min_lon = normalization_stats['cdr_min_lon']
            self.cdr_max_lon = normalization_stats['cdr_max_lon']
            self.cdr_min_lat = normalization_stats['cdr_min_lat']
            self.cdr_max_lat = normalization_stats['cdr_max_lat']
            self.road_gps_min_lon = normalization_stats['road_gps_min_lon']
            self.road_gps_max_lon = normalization_stats['road_gps_max_lon']
            self.road_gps_min_lat = normalization_stats['road_gps_min_lat']
            self.road_gps_max_lat = normalization_stats['road_gps_max_lat']
        else:
            # 使用默认值作为后备
            print("警告: 未提供标准化统计信息，使用默认值")
            self.gps_mean_lon = 104.1
            self.gps_mean_lat = 30.65
            self.gps_std_lon = 0.3
            self.gps_std_lat = 0.15
            self.cdr_mean_lon = 104.1
            self.cdr_mean_lat = 30.65
            self.cdr_std_lon = 0.3
            self.cdr_std_lat = 0.15
            self.road_gps_mean_lon = 104.1
            self.road_gps_mean_lat = 30.65
            self.road_gps_std_lon = 0.3
            self.road_gps_std_lat = 0.15
            
            

            
            # 添加默认的0-1标准化参数
            self.gps_min_lon = -180.0
            self.gps_max_lon = 180.0
            self.gps_min_lat = -90.0
            self.gps_max_lat = 90.0
            self.cdr_min_lon = -180.0
            self.cdr_max_lon = 180.0
            self.cdr_min_lat = -90.0
            self.cdr_max_lat = 90.0
            self.road_gps_min_lon = -180.0
            self.road_gps_max_lon = 180.0
            self.road_gps_min_lat = -90.0
            self.road_gps_max_lat = 90.0
            


        
        # Output Projection layer
        self.share_output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)

        )
        # Input Projection layer
        self.share_input_projection = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        # Contrastive parameter layer
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1.0 / contrastive_temperature), dtype=torch.float32,requires_grad=True)
        )
        self.logit_scale_max = 100.0  
        self.logit_scale_min = 0.0     
        
        # GPS、Road和CDR序列编码器
        self.gps_encoder = SequentialEncoder(
            input_dim=2, d_model=d_model, nhead=nhead, 
            num_layers=num_encoder_layers, dropout=dropout
        )
        
        self.cdr_encoder = SequentialEncoder(
            input_dim=2, d_model=d_model, nhead=nhead,
            num_layers=num_encoder_layers, dropout=dropout
        )
        self.road_segment_encoder = SequentialEncoder(
            input_dim=2, d_model=d_model, nhead=nhead,
            num_layers=num_encoder_layers, dropout=dropout
        )
        

        
        # mask
        self.mlm_prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)  # 中间层，不直接输出最终结果
        )
        self.gps_cdr_prediction_head = nn.Linear(d_model, 2)  # 经纬度预测



        
        # 多模态融合模块
        self.modal_fusion = EnhancedMultiModalFusion(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        
        # 最终序列生成解码器
        self.final_generator = SequenceGenerator(
            d_model=d_model, nhead=nhead, num_layers=num_decoder_layers,
            road_network_size=self.road_network_size, dropout=dropout,
            generation_mode=self.generation_mode)
        


        # 数值稳定性参数
        self.eps = 1e-8
        self.clip_value = 10.0


        self.time_encoder = TimeEncoding(d_model=d_model)


        
        # 初始化权重
        self.apply(self._init_weights)
    
    def set_generation_mode(self, mode):
        """设置生成模式
        
        Args:
            mode: 'parallel' 或 'autoregressive'
        """
        if mode not in ['parallel', 'autoregressive']:
            raise ValueError(f"不支持的生成模式: {mode}，请选择 'parallel' 或 'autoregressive'")
        
        self.generation_mode = mode
        # 同时更新所有生成器的模式
        if hasattr(self, 'intermediate_generator'):
            self.intermediate_generator.generation_mode = mode
        if hasattr(self, 'final_generator'):
            self.final_generator.generation_mode = mode
        
        print(f"生成模式已切换为: {mode}")
        
    def _deterministic_randn(self, *size, device=None, dtype=torch.float32):
        """生成确定性的随机张量"""
        return torch.randn(*size, device=device, dtype=dtype, generator=self.deterministic_generator)
    
    def _init_weights(self, module):
        """改进的权重初始化 - 使用确定性随机数"""
        if isinstance(module, nn.Linear):
            # 使用确定性种子初始化
            with torch.no_grad():
                nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            with torch.no_grad():
                nn.init.normal_(module.weight, mean=0.0, std=0.005)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)
        elif hasattr(module, 'weight') and module.weight.dim() > 1:
            with torch.no_grad():
                nn.init.xavier_uniform_(module.weight, gain=0.02)
        
    def _safe_tensor_operation(self, tensor, operation_name):
        """安全的张量操作包装器"""
        if tensor is None:
            return None
            
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"警告：{operation_name} 输入包含NaN或无穷大，进行修复")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        tensor = torch.clamp(tensor, min=-self.clip_value, max=self.clip_value)
        return tensor
        
    def forward(self, gps_data=None, cdr_data=None, road_seg_data=None, 
                road_network=None, target=None, **kwargs):
        
        batch_size = self._get_batch_size(gps_data, cdr_data, road_seg_data, target)
        device = self.device
        
        # 提取扩展数据
        road_time_data = kwargs.get('road_time_data',None)
        gps_time_data = kwargs.get('gps_time_data',None)
        cdr_time_data = kwargs.get('cdr_time_data',None)
        original_cpath_data = kwargs.get('original_cpath_data', target)  # 使用original_cpath_data作为最终target
        

        gps_data = self._robust_preprocess_input_data(gps_data, "GPS")     
        cdr_data = self._robust_preprocess_input_data(cdr_data, "CDR") 
        road_seg_data = self._robust_preprocess_input_data(road_seg_data, "ROAD_GPS")

        road_time_data = self._robust_preprocess_time_data(road_time_data)
        gps_time_data = self._robust_preprocess_time_data(gps_time_data)
        cdr_time_data = self._robust_preprocess_time_data(cdr_time_data)

        original_cpath_data = self._robust_preprocess_road_data(original_cpath_data)

     
        gps_data_padding = (gps_data[:, :, 0] != -1.0) & (gps_data[:, :, 1] != -1.0)# 获取GPS数据的掩码（填充值为-1.0）   
        cdr_data_padding = (cdr_data[:, :, 0] != -1.0) & (cdr_data[:, :, 1] != -1.0)# 获取CDR数据的掩码（填充值为-1.0）
        road_data_padding = (road_seg_data[:, :, 0] != -1.0) & (road_seg_data[:, :, 1] != -1.0)        # 获取路段数据的掩码（填充值为-1）


        
        
        # 1. 编码各种模态 + MLM任务
        encoded_embeddings = {}
        time_positions={}
        time_embeddings = {}
        contrastive_embeddings = {}   
        
        try:
            # GPS序列编码 + MLM
            if gps_data is not None:
                
                # 创建MLM掩码
                masked_gps_data, original_gps_data, masked_gps_time_data, original_gps_time_data, gps_mask = self._create_mlm_mask(gps_data, gps_time_data, "GPS", gps_data_padding)
               
                # GPS编码 + 时间编码
                time_positions['gps']=gps_time_data
                gps_time_embedding = self.time_encoder(gps_time_data)

                gps_embedding,gps_mlm_embedding,gps_traj_rep=self.gps_encoder(masked_gps_data,gps_time_embedding,~gps_data_padding,self.share_input_projection,self.share_output_projection)  
         
                gps_embedding = self._safe_tensor_operation(gps_embedding, "GPS编码")
                gps_mlm_embedding=self._safe_tensor_operation(gps_mlm_embedding,"GPS MLM编码")
                
                encoded_embeddings['gps'] = gps_embedding

                contrastive_embeddings['gps'] = gps_traj_rep
                
            # CDR序列编码 + MLM
            if cdr_data is not None:

                # 创建MLM掩码
                masked_cdr_data, original_cdr_data, masked_cdr_time_data, original_cdr_time_data, cdr_mask = self._create_mlm_mask(cdr_data, cdr_time_data, "CDR", cdr_data_padding)


                # CDR编码 + 时间编码
                time_positions['cdr']=cdr_time_data
                cdr_time_embedding = self.time_encoder(cdr_time_data)

                cdr_embedding,cdr_mlm_embedding,cdr_traj_rep=self.cdr_encoder(masked_cdr_data,cdr_time_embedding,~cdr_data_padding,self.share_input_projection,self.share_output_projection)

                cdr_embedding = self._safe_tensor_operation(cdr_embedding, "CDR编码")
                cdr_mlm_embedding=self._safe_tensor_operation(cdr_mlm_embedding,"CDR MLM编码")
                
                encoded_embeddings['cdr'] = cdr_embedding
                contrastive_embeddings['cdr'] = cdr_traj_rep
                
                
            # 路段序列编码 + MLM
            if road_seg_data is not None:

                # 创建MLM掩码
                masked_road_data, original_road_data, masked_road_time_data, original_road_time_data, road_mask = self._create_mlm_mask(road_seg_data, road_time_data, "ROAD_GPS", road_data_padding)

                # 路段编码 + 时间编码
                time_positions['road_seg']=road_time_data
                road_time_embedding = self.time_encoder(road_time_data)

                road_seg_embedding,road_seg_mlm_embedding,road_traj_rep=self.road_segment_encoder(masked_road_data,road_time_embedding,~road_data_padding,self.share_input_projection,self.share_output_projection)

                road_seg_embedding = self._safe_tensor_operation(road_seg_embedding, "路段编码")
                road_seg_mlm_embedding=self._safe_tensor_operation(road_seg_mlm_embedding,"路段 MLM编码")

                encoded_embeddings['road_seg'] = road_seg_embedding
                contrastive_embeddings['road_seg'] = road_traj_rep
                
                
        except Exception as e:
            print(f"编码阶段发生错误: {e}")
            
            return self._default_output(batch_size, device, original_cpath_data)
        
        # 如果没有任何输入，返回默认结果
        if not encoded_embeddings:
            return self._default_output(batch_size, device, original_cpath_data)
        
        try:
            # 2. 多模态融合
            fusion_inputs = {}
            fusion_padding_mask = {}
            fusion_time_positions = {}
            fusion_time_embeddings = None
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

                      

            fused_representation,fused_representation_padding_mask = self.modal_fusion(fusion_inputs,fusion_time_embeddings,fusion_padding_mask,fusion_time_positions)
            fused_representation = self._safe_tensor_operation(fused_representation, "模态融合")
            
            # 3. 最终生成：fused_representation 生成 original_cpath_data
            final_sequence, final_generation_info = self.final_generator(
                fused_representation, fused_representation_padding_mask=fused_representation_padding_mask, target=original_cpath_data
            )
            final_sequence = self._safe_tensor_operation(final_sequence, "最终序列生成")
            
        except Exception as e:
            print(f"推理阶段发生错误: {e}")
            return self._default_output(batch_size, device, original_cpath_data)
        
        # 4. 计算损失
        if original_cpath_data is not None: 

            
            try:
                loss_dict = self.compute_comprehensive_loss_with_generation(
                    final_sequence, original_cpath_data, {},
                    final_generation_info, contrastive_embeddings
                )
            except Exception as e:
                print(f"损失计算发生错误: {e}")
                loss_dict = {"total_loss": torch.tensor(1.0, device=device, requires_grad=True)}
            return final_sequence, loss_dict, None  # 不再返回中间任务logits
        
        return final_sequence, {"total_loss": torch.tensor(0.0, device=device, requires_grad=True)}, None


    def _robust_preprocess_input_data(self, data, data_type):
        """更稳健的输入数据预处理"""
        if data is None:
            return None
            
        # 检查和修复异常值
        if torch.isnan(data).any() or torch.isinf(data).any():
            print(f"修复：{data_type}输入数据包含NaN或无穷大")
            data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if data_type in ["GPS", "CDR","ROAD_GPS"]:
            # 对于经纬度数据，使用更稳定的预处理方式
            data = torch.clamp(data, min=-180.0, max=180.0)
            
            # 使用0-1标准化参数
            if data_type == "GPS":
                min_lon, min_lat = self.gps_min_lon, self.gps_min_lat
                max_lon, max_lat = self.gps_max_lon, self.gps_max_lat
            elif data_type == "CDR":  # CDR
                min_lon, min_lat = self.cdr_min_lon, self.cdr_min_lat
                max_lon, max_lat = self.cdr_max_lon, self.cdr_max_lat
            elif data_type == "ROAD_GPS":
                
                min_lon, min_lat = self.road_gps_min_lon, self.road_gps_min_lat
                max_lon, max_lat = self.road_gps_max_lon, self.road_gps_max_lat
            else:
                print(f"警告：未知的数据类型 {data_type}，无法确定标准化参数")
                exit(1)
            
            # 计算范围，避免除零
            range_lon = max_lon - min_lon
            range_lat = max_lat - min_lat

            if range_lon == 0 or range_lat == 0:
                print(f"修复：{data_type}输入数据经度或纬度范围为0，无法标准化")
                return data  # 返回原始数据，避免除零错误
            
            # 分别标准化经度和纬度
            # 只有当数据不为-1.0(填充)时才进行标准化
            valid_lat_mask = data[:, :, 0] != -1.0
            valid_lon_mask = data[:, :, 1] != -1.0

            # 使用正确的布尔索引方式 - 为每个位置创建完整的索引
            batch_indices_lat, seq_indices_lat = torch.where(valid_lat_mask)
            batch_indices_lon, seq_indices_lon = torch.where(valid_lon_mask)
            
            # 使用0-1标准化公式: (x - min) / (max - min)
            data[batch_indices_lat, seq_indices_lat, 0] = (data[batch_indices_lat, seq_indices_lat, 0] - min_lat) / range_lat
            data[batch_indices_lon, seq_indices_lon, 1] = (data[batch_indices_lon, seq_indices_lon, 1] - min_lon) / range_lon
            
            # 最终限制范围到[0, 1]
            data = torch.clamp(data, min=0.0, max=1.0)
        
        return data   
    
    def _robust_preprocess_road_data(self, data):
        """更稳健的路段数据预处理"""
        if data is None:
            return None
            
        # 检查异常值
        if torch.isnan(data.float()).any() or torch.isinf(data.float()).any():
            print("修复：路段数据包含NaN或无穷大")
            data = torch.nan_to_num(data.float(), nan=0, posinf=self.road_network_size-1, neginf=0).long()
        
        # 确保路段ID在有效范围内
        data = torch.clamp(data, min=-1, max=self.road_network_size)
        
        return data.to(torch.long)
    
    def _robust_preprocess_time_data(self, data):   
        """更稳健的时间数据预处理"""
        if data is None:
            return None
            
        # 检查异常值
        if torch.isnan(data.float()).any() or torch.isinf(data.float()).any():
            print("修复：时间数据包含NaN或无穷大")
            data = torch.nan_to_num(data.float(), nan=0, posinf=1.0, neginf=0).long()

        # 归一化时间特征
        #data[:,:,0]=data[:,:,0] / 12  # 月份归一化到[0,1]
        #data[:,:,1]=data[:,:,1] / 7   # 星期归一化到[0,1]
        #data[:,:,2]=data[:,:,2] / 24  # 小时归一化到[0,1]
        #data[:,:,3]=data[:,:,3] / 60  # 分钟归一化到[0,1]
        
        # 确保时间在有效范围内
        data = torch.clamp(data, min=-1.0, max=60)
        
        return data.to(torch.long)
    
    def _get_batch_size(self, *tensors):
        for tensor in tensors:
            if tensor is not None:
                return tensor.size(0)
        return 1
    
    def _default_output(self, batch_size, device, target):
        """返回安全的默认输出 - 使用确定性的零张量"""
        if target is not None:
            seq_len = target.size(1)
        else:
            seq_len = 50
        
        # 使用确定性的零张量而不是随机张量
        pred = torch.log_softmax(
            torch.zeros(batch_size, seq_len, self.road_network_size, device=device),
            dim=-1
        )
        loss = {"total_loss": torch.tensor(1.0, device=device, requires_grad=True)}
        intermediate_logits = torch.log_softmax(
            torch.zeros(batch_size, seq_len, self.road_network_size, device=device),
            dim=-1
        )
        return pred, loss, intermediate_logits

    def _create_mlm_mask(self, input_data,time_data, data_type, data_padding=None):
        """创建MLM掩码"""
        if input_data is None:
            return None, None, None
            
        batch_size, seq_len = input_data.shape[:2]
        device = self.device

        # 设备检查（已修复）
        if self.deterministic_generator.device != device:
            print(f"device为 {device}")
            print(f"deterministic_generator 设备为 {self.deterministic_generator.device}")
        
        # 如果没有提供掩码，假设所有位置都有效
        if data_padding is None:
            data_padding = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # 1. 批量生成随机掩码（向量化操作）
        mask_matrix = torch.rand(batch_size, seq_len, generator=self.deterministic_generator, device=device) < self.mlm_mask_prob
        
        # 2. 批量保护关键位置（向量化操作）
        # 保护填充位置
        mask_matrix = mask_matrix & data_padding
        
        # 保护起点和终点
        valid_indices = []
        for i in range(batch_size):
            indices = torch.where(data_padding[i])[0]
            valid_indices.append(indices)
            if len(indices) > 0:
                mask_matrix[i, indices[0]] = False  # 保护起点
                mask_matrix[i, indices[-1]] = False  # 保护终点
        
        
        # 3. 批量应用掩码策略（向量化操作）
        original_data = input_data.clone()
        masked_data = input_data.clone()


        original_time_data = time_data.clone().to(torch.long)
        masked_time_data = time_data.clone().to(torch.long)

        #print("随机掩码",mask_matrix[1,:])
        #print("填充掩码",data_padding[1,:])

        # 批量生成随机策略
        strategy_rand = torch.rand(batch_size, seq_len, generator=self.deterministic_generator, device=device)
        
        # 替换掩码（< mlm_replace_prob）
        replace_mask = mask_matrix & (strategy_rand < self.mlm_replace_prob)
        if data_type in ["GPS", "CDR","ROAD_GPS"]:
            # 修复：为每个被掩码位置创建正确的形状
            num_replace = replace_mask.sum().item()
            if num_replace > 0:

                # 创建正确形状的替换值
                replace_values = torch.full((num_replace, 2), -999.0, device=device)
                replace_time_values = torch.full((num_replace, ), -999, device=device, dtype=torch.long)

                masked_data[replace_mask] = replace_values
                mask_indices = replace_mask.nonzero(as_tuple=True)

                
                masked_time_data[mask_indices[0], mask_indices[1], 2] = replace_time_values
                masked_time_data[mask_indices[0], mask_indices[1], 3] = replace_time_values

        else:
            # 路段数据
            
            num_replace = replace_mask.sum().item()
            if num_replace > 0:
                replace_values = torch.full((num_replace,), -999, device=device, dtype=torch.long)
                replace_time_values = torch.full((num_replace, ), -999, device=device, dtype=torch.long)
                masked_data[replace_mask] = replace_values
                mask_indices = replace_mask.nonzero(as_tuple=True)
                masked_time_data[mask_indices[0], mask_indices[1], 2] = replace_time_values
                masked_time_data[mask_indices[0], mask_indices[1], 3] = replace_time_values
        
        # 随机替换掩码（mlm_replace_prob <= x < mlm_replace_prob + mlm_random_prob）
        random_mask = mask_matrix & (strategy_rand >= self.mlm_replace_prob) & (strategy_rand < self.mlm_replace_prob + self.mlm_random_prob)
        if data_type in ["GPS", "CDR","ROAD_GPS"]:

            # 批量生成随机值
            num_random = random_mask.sum().item()
            if num_random > 0:
                random_values = torch.randn(num_random, 2, device=device) * 0.1
                masked_data[random_mask] = random_values

                # 生成合理的随机小时
                random_hours = torch.randint(0, 24, (num_random,), device=device)
                random_minutes = torch.randint(0, 60, (num_random,), device=device)

                # 只修改最后两个维度（小时和分钟）
                random_indices = random_mask.nonzero(as_tuple=True)

                masked_time_data[random_indices[0], random_indices[1], 2] = random_hours
                masked_time_data[random_indices[0], random_indices[1], 3] = random_minutes

        else:
            
            # 批量生成随机路段ID
            num_random = random_mask.sum().item()
            if num_random > 0:
                random_ids = torch.randint(0, self.road_network_size, (num_random,), device=device)
                masked_data[random_mask] = random_ids
                # 生成合理的随机小时（1-24）和分钟（1-60）
                random_hours = torch.randint(0, 24, (num_random,), device=device)
                random_minutes = torch.randint(0, 60, (num_random,), device=device)
                
                # 只修改最后两个维度（小时和分钟）
                random_indices = random_mask.nonzero(as_tuple=True)
                masked_time_data[random_indices[0], random_indices[1], 2] = random_hours
                masked_time_data[random_indices[0], random_indices[1], 3] = random_minutes
        
        # 保持原样的掩码位置自动处理（mask_matrix中对应位置为False）
        #print("最后返回的掩码",mask_matrix[1,:])
        return masked_data, original_data, masked_time_data, original_time_data, mask_matrix
    
    def compute_comprehensive_loss_with_generation(self, pred_sequence, target, 
                                           mlm_results, final_generation_info, 
                                           contrastive_embeddings=None):
        """计算包含主任务、MLM任务、对比学习、序列正则化的综合损失"""
        device = self.device
        
        try:
            if pred_sequence.numel() == 0 or target.numel() == 0:
                return {'total_loss': torch.tensor(5.0, device=device, requires_grad=True)}
            
            if torch.isnan(pred_sequence).any() or torch.isinf(pred_sequence).any():
                return {'total_loss': torch.tensor(5.0, device=device, requires_grad=True)}
            
            # 1. 主要损失：最终生成任务（fused -> original_cpath）
            target_clamped = torch.clamp(target.long(), min=-1, max=self.road_network_size-1)
            pred_sequence_clamped = torch.clamp(pred_sequence, min=-10, max=10)
            
            pred_flat = pred_sequence_clamped.view(-1, self.road_network_size)
            target_flat = target_clamped.view(-1)
            
            main_loss = F.cross_entropy(
                pred_flat, target_flat, 
                ignore_index=-1, 
                reduction='mean',
                label_smoothing=0.1
            )
            main_loss = torch.clamp(main_loss, min=1e-4, max=15.0)
            
            # 2. MLM任务损失(已屏蔽)
            mlm_losses = {}
            total_mlm_loss = torch.tensor(0.0, device=device, requires_grad=True)
            time_mlm_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            for modality_name, mlm_data in mlm_results.items():
                try:
                    modality_loss = self._compute_mlm_loss(
                        mlm_data['embedding'], 
                        mlm_data['original_data'], 
                        mlm_data['mask'], 
                        mlm_data['data_type']
                    )
                    mlm_losses[f'{modality_name}_mlm_loss'] = modality_loss.item()
                    if modality_name in ['gps_time', 'cdr_time', 'road_time']:
                        time_mlm_loss = time_mlm_loss + modality_loss
                    else:
                        total_mlm_loss = total_mlm_loss + modality_loss
                except Exception as e:
                    print(f"MLM任务 {modality_name} 损失计算错误: {e}")
                    mlm_losses[f'{modality_name}_mlm_loss'] = 0.0
            total_mlm_loss = total_mlm_loss + time_mlm_loss  # 时间损失平均分配
            
            # 3. 序列级别正则化
            sequence_reg = self._compute_sequence_regularization(pred_sequence, target)

            # 4. 对比学习损失 
            contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)        
            if contrastive_embeddings is not None and self.contrastive_loss_weight > 0:
                contrastive_loss = self._compute_contrastive_loss(contrastive_embeddings)

            # 5.可学习温度参数监控
            logit_scale = 1/self.logit_scale.exp()
   
            
            # 6. 总损失（主任务+MLM任务+对比学习+序列正则化）
            total_loss = (self.main_loss_weight * main_loss + 
                         self.mlm_loss_weight * total_mlm_loss + 
                         self.contrastive_loss_weight * contrastive_loss +
                         self.sequence_reg_weight * sequence_reg)
            
            loss_dict = {
                'total_loss': total_loss,
                'main_loss': main_loss.item() if hasattr(main_loss, 'item') else float(main_loss),
                'mlm_loss': total_mlm_loss.item() if hasattr(total_mlm_loss, 'item') else 0.0,
                'contrastive_loss': contrastive_loss.item() if hasattr(contrastive_loss, 'item') else 0.0,
                'contrastive_temperature': logit_scale,
                'sequence_reg': sequence_reg.item() if hasattr(sequence_reg, 'item') else 0.0,
            }
            
            # 添加详细的MLM任务损失
            loss_dict.update(mlm_losses)
            
            return loss_dict
            
        except Exception as e:
            return {
                'total_loss': torch.tensor(5.0, device=device, requires_grad=True),
                'main_loss': 5.0,
                'mlm_loss': 0.0,
                'contrastive_loss': 0.0,
                'contrastive_temperature': 0.0,
                'sequence_reg': 0.0
            }
    
    def _compute_contrastive_loss(self, contrastive_embeddings):
        """计算对比学习损失"""
        try:
            device = self.device
            batch_size = len(contrastive_embeddings)
            
            if batch_size < 2:
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            # 投影到对比学习空间
            projected_embeddings = []
            for modality_name, embedding in contrastive_embeddings.items():
                if embedding is not None:
                    #projected = self.contrastive_projection(embedding)
                    projected = embedding
                    projected_embeddings.append(projected)
            
            if len(projected_embeddings) < 2:
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            # 构建正负样本对
            contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
            num_pairs = 0

            logit_scale_clipped = torch.clamp(self.logit_scale, self.logit_scale_min, self.logit_scale_max)
            scale = logit_scale_clipped.exp()  # = 1 / τ
            
            # 多模态对比：不同模态的相同样本作为正样本对
            for i in range(len(projected_embeddings)):
                for j in range(i+1, len(projected_embeddings)):
                    anchor = projected_embeddings[i]
                    positive = projected_embeddings[j]
                    
                    # 计算相似度矩阵
                    similarity_matrix = torch.matmul(
                        F.normalize(anchor, dim=-1), 
                        F.normalize(positive, dim=-1).transpose(0, 1)
                    ) * scale #/ self.contrastive_temperature  # 使用温度参数调整相似度
                    
                    # 计算InfoNCE损失
                    labels = torch.arange(anchor.size(0), device=device)
                    loss_i = F.cross_entropy(similarity_matrix, labels)
                    loss_j = F.cross_entropy(similarity_matrix.transpose(0, 1), labels)
                    
                    contrastive_loss = contrastive_loss + (loss_i + loss_j) / 2
                    num_pairs += 1
            

            if num_pairs > 0:
                #print(f"对比学习计算了 {num_pairs} 对模态")
                contrastive_loss = contrastive_loss/ num_pairs 
            else:
                contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            return torch.clamp(contrastive_loss, min=1e-4, max=30.0)
            
        except Exception as e:
            print(f"对比学习损失计算错误: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)   

    def _compute_mlm_loss(self, embeddings, original_data, mask_matrix, data_type):
        """计算MLM损失 - 修复版本"""
        if embeddings is None or original_data is None or mask_matrix is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        batch_size, seq_len = embeddings.shape[:2]
        
        if not mask_matrix.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        masked_positions = mask_matrix.nonzero(as_tuple=True)
        if len(masked_positions[0]) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 批量处理掩码位置
        masked_embeddings = embeddings[masked_positions[0], masked_positions[1]]
        original_values = original_data[masked_positions[0], masked_positions[1]]
        
        # 批量预测（向量化操作）
        shared_features = self.mlm_prediction_head(masked_embeddings)
        #shared_features = masked_embeddings
        
        if data_type in ["GPS", "CDR","ROAD_GPS"]:
            predictions = self.gps_cdr_prediction_head(shared_features)
            if original_values.dim() == 1:
                original_values = original_values.view(-1, 1)
                if predictions.dim() == 2 and predictions.size(1) == 2:
                    original_values = original_values.expand(-1, 2)

            loss = F.mse_loss(predictions, original_values, reduction='mean')*100


        else:
            predictions = self.road_prediction_head(shared_features)
            original_values = original_values.long()


            # 批量交叉熵损失
            loss = F.cross_entropy(
                predictions, 
                original_values,
                ignore_index=self.road_network_size,
                reduction='mean'
            )
        
        return torch.clamp(loss, min=1e-4, max=15.0)
    
    def _compute_sequence_regularization(self, pred_sequence, target):
        """计算序列级别的正则化"""
        try:
            batch_size, seq_len, vocab_size = pred_sequence.shape
            device = pred_sequence.device
            
            if seq_len <= 1:
                return torch.tensor(0.0, device=device)
            
            # 计算相邻预测的平滑性
            pred_probs = F.softmax(pred_sequence, dim=-1)
            
            # 计算相邻时间步预测分布的KL散度
            smoothness_loss = 0.0
            for i in range(seq_len - 1):
                curr_dist = pred_probs[:, i, :]
                next_dist = pred_probs[:, i+1, :]
                
                # 计算KL散度
                kl_div = F.kl_div(
                    torch.log(next_dist + 1e-8), 
                    curr_dist, 
                    reduction='batchmean'
                )
                smoothness_loss += kl_div
            
            return smoothness_loss / (seq_len - 1)
            
        except Exception:
            return torch.tensor(0.0, device=pred_sequence.device)
    
    @property
    def device(self):
        return next(self.parameters()).device


