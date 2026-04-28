


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

# 设置线程数
torch.set_num_threads(5)

# 设置CUDA设备
dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
if torch.cuda.is_available():
    torch.cuda.set_device(dev_id)


from dataloader import get_trifusion_loaders


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


def get_seq_emb_from_traj_withALLModel(model, test_data, batch_size=64):
    """从轨迹数据中提取序列嵌入（fused_representation）"""
    device = next(model.parameters()).device
    model.eval()
    
    # 解包测试数据
    (route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat,
     grid_data, masked_grid_assign_mat, gps_data_grid, masked_gps_assign_mat_grid, grid_assign_mat,
     gps_length, gps_length_grid, dataset) = test_data
    
    num_samples = len(dataset)
    print(f"处理 {num_samples} 个样本...")
    
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_indices = list(range(i, batch_end))
            
            # 准备批次数据
            batch_route_data = route_data[batch_indices] if route_data is not None else None
            batch_gps_data = gps_data[batch_indices] if gps_data is not None else None
            batch_grid_data = grid_data[batch_indices] if grid_data is not None else None
            batch_gps_data_grid = gps_data_grid[batch_indices] if gps_data_grid is not None else None
            
            # 移动到设备
            if batch_route_data is not None:
                batch_route_data = batch_route_data.to(device)
            if batch_gps_data is not None:
                batch_gps_data = batch_gps_data.to(device)
            if batch_grid_data is not None:
                batch_grid_data = batch_grid_data.to(device)
            if batch_gps_data_grid is not None:
                batch_gps_data_grid = batch_gps_data_grid.to(device)
            
            try:
                # 构建模型输入
                batch_size_actual = len(batch_indices)
                
                # 使用模型的各个编码器提取特征并融合
                encoded_embeddings = {}
                
                # GPS序列编码
                if batch_gps_data is not None:
                    gps_embedding = model.gps_encoder(batch_gps_data)
                    encoded_embeddings['gps'] = gps_embedding
                
                # CDR序列编码（如果有CDR数据的话，这里使用GPS grid数据代替）
                if batch_gps_data_grid is not None:
                    cdr_embedding = model.cdr_encoder(batch_gps_data_grid)
                    encoded_embeddings['cdr'] = cdr_embedding
                
                # 路段序列编码
                if batch_route_data is not None:
                    road_seg_embedding = model.road_segment_encoder(batch_route_data)
                    encoded_embeddings['road_seg'] = road_seg_embedding
                
                # 多模态融合
                if encoded_embeddings:
                    fused_representation = model.modal_fusion(encoded_embeddings)
                    
                    # 取序列的平均池化作为整体表示
                    if fused_representation.dim() == 3:  # (batch, seq_len, feature_dim)
                        # 简单平均池化
                        fused_representation = torch.mean(fused_representation, dim=1)
                    
                    embeddings.append(fused_representation.cpu())
                else:
                    # 如果没有有效的输入，创建零向量
                    zero_embedding = torch.zeros(batch_size_actual, model.d_model)
                    embeddings.append(zero_embedding)
                    
            except Exception as e:
                print(f"批次 {i//batch_size} 处理失败: {e}")
                # 创建零向量作为备用
                zero_embedding = torch.zeros(batch_size_actual, model.d_model)
                embeddings.append(zero_embedding)
    
    # 拼接所有嵌入
    seq_embedding = torch.cat(embeddings, dim=0)
    print(f"提取的嵌入形状: {seq_embedding.shape}")
    
    return seq_embedding


def data_prepare(task_data, padding_id):
    """准备时间估计任务的数据"""
    min_len, max_len = 10, 256

    split_df = task_data  # 'cpath_list', 'total_time'
    split_df['path_len'] = split_df['cpath_list'].map(len)
    split_df = split_df.loc[(split_df['path_len'] > min_len) & (split_df['path_len'] < max_len)]

    num_samples = len(split_df)
    x_arr = np.full([num_samples, max_len], padding_id, dtype=np.int32)
    y_arr = np.zeros([num_samples], dtype=np.float32)

    for i in range(num_samples):
        row = split_df.iloc[i]
        path_arr = np.array(row['cpath_list'], dtype=np.int32)
        x_arr[i, :row['path_len']] = path_arr
        y_arr[i] = row['total_time']

    return torch.LongTensor(x_arr), torch.FloatTensor(y_arr), split_df['path_len'].values


def evaluation_with_fused_representation(seq_embedding, task_data, num_nodes, fold=5):
    """使用融合表示进行时间估计评估"""
    batch_size = 64
    
    # 准备标签数据（从task_data中获取total_time）
    if 'total_time' not in task_data.columns:
        raise ValueError("数据中缺少total_time字段，无法进行时间估计")
    
    y = torch.FloatTensor(task_data['total_time'].values)
    x = seq_embedding
    
    if len(x) != len(y):
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        print(f"警告：特征和标签长度不匹配，截断到 {min_len}")
    
    split = x.shape[0] // fold
    print(f"使用{fold}折交叉验证，每折{split}个样本，总共{x.shape[0]}个样本")

    device_flag = True
    fold_preds = []
    fold_trues = []
    
    for i in range(fold):
        eval_idx = list(range(i * split, (i + 1) * split, 1))
        train_idx = list(set(list(range(x.shape[0]))) - set(eval_idx))

        x_train, x_eval = x[train_idx], x[eval_idx]
        y_train, y_eval = y[train_idx], y[eval_idx]

        fold_trues.append(y_eval)

        # 标准化标签
        y_train, mean, std = label_norm(y_train)
        
        # 初始化模型
        model = MLPReg(x.shape[1], 3, nn.ReLU()).cuda()

        if device_flag:
            print('device: ', next(model.parameters()).device)
            device_flag = False

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 训练参数
        patience = 3
        epoch_threshold = 10
        epoch_num = 50
        best_epoch = 0
        best_mae = 1e9
        best_rmse = 1e9
        best_preds = None
        
        for epoch in range(1, epoch_num + 1):
            model.train()
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
            mae = mean_absolute_error(y_eval, y_preds)
            rmse = mean_squared_error(y_eval, y_preds) ** 0.5
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
    mae = mean_absolute_error(y_trues, y_preds)
    rmse = mean_squared_error(y_trues, y_preds) ** 0.5
    print(f'travel time estimation | MAE: {mae:.4f}, RMSE: {rmse:.4f}')

    return best_epoch, best_mae, best_rmse


def evaluation(city, exp_path, model_name, start_time):
    """主评估函数"""
    route_min_len, route_max_len = 10, 100
    gps_min_len, gps_max_len = 10, 256
    grid_min_len, grid_max_len = 10, 100
    
    model_path = os.path.join(exp_path, model_name)

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")

    # 准备序列任务数据
    test_seq_data = pickle.load(
        # open('../{}/{}_eval_extension3.pkl'.format(city, city), 'rb'))
        open('/home/hzx/xy/traj/MVTraj/{}/{}_eval_extension3.pkl'.format(city, city), 'rb'))
    
    
    # 采样数据（如果数据量太大）
    if len(test_seq_data) > 50000:
        test_seq_data = test_seq_data.sample(50000, random_state=0)

    print(f"评估数据样本数：{len(test_seq_data)}")

    # 构建road_network（参考dataloader的逻辑）
    # 从数据中提取所有唯一路段ID
    unique_segments = set()
    
    # 添加所有路段ID字段
    for col in ['monitor_opath_list', 'sparse_gps2route_list', 'cdr_gps2route_list', 
               'merged_route_list', 'original_cpath_list']:
        if col in test_seq_data.columns:
            road_seg_list = test_seq_data[col].tolist()
            for segs in road_seg_list:
                if segs:  # 检查非空
                    unique_segments.update(segs)
    
    # 提取路段连接关系
    road_connections = []
    for col in ['monitor_opath_list', 'merged_route_list', 'original_cpath_list']:
        if col in test_seq_data.columns:
            road_seg_list = test_seq_data[col].tolist()
            for segs in road_seg_list:
                if segs:  # 检查非空
                    for i in range(len(segs) - 1):
                        road_connections.append((segs[i], segs[i+1]))
    
    # 去重
    road_connections = list(set(road_connections))
    
    # 创建road_network（使用dataloader中的prepare_road_network函数）
    from dataloader import prepare_road_network
    road_network = prepare_road_network(
        list(unique_segments), road_connections
    )
    
    # 获取road_network_size（参考dataloader的方式）
    road_network_size = road_network.x.size(0)
    print("road_network_size:", road_network_size)

    # 加载训练好的模型
    print(f"加载模型：{model_path}")
    checkpoint = torch.load(model_path, map_location=f"cuda:{dev_id}")
    seq_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # 如果checkpoint是完整的字典，需要重新构建模型
    if isinstance(seq_model, dict):
        # 重新构建模型
        seq_model = OptimizedTriFusionPathNet(
            d_model=256,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            road_network_size=road_network_size,
            dropout=0.1,
            generation_mode='parallel'
        ).to(f"cuda:{dev_id}")
        seq_model.load_state_dict(checkpoint['model_state_dict'])
    
    seq_model.eval()

    # 准备数据用于提取嵌入
    # 这里需要构造compatible的数据格式
    route_length = test_seq_data['route_length'].values if 'route_length' in test_seq_data.columns else None
    
    # 构造数据元组（简化版本，根据实际数据调整）
    route_data = None
    gps_data = None 
    grid_data = None
    gps_data_grid = None
    
    # 尝试从数据中提取路段数据
    if 'monitor_opath_list' in test_seq_data.columns:
        route_lists = test_seq_data['monitor_opath_list'].tolist()
        max_len = min(route_max_len, max([len(r) for r in route_lists]))
        route_data = torch.zeros((len(route_lists), max_len), dtype=torch.long)
        for i, route in enumerate(route_lists):
            seq_len = min(len(route), max_len)
            route_data[i, :seq_len] = torch.tensor(route[:seq_len])
    
    # 尝试从数据中提取GPS数据
    if 'sparse_lat_list' in test_seq_data.columns and 'sparse_lng_list' in test_seq_data.columns:
        lat_lists = test_seq_data['sparse_lat_list'].tolist()
        lng_lists = test_seq_data['sparse_lng_list'].tolist()
        max_len = min(gps_max_len, max([max(len(lat), len(lng)) for lat, lng in zip(lat_lists, lng_lists)]))
        gps_data = torch.zeros((len(lat_lists), max_len, 2), dtype=torch.float32)
        for i, (lats, lngs) in enumerate(zip(lat_lists, lng_lists)):
            seq_len = min(len(lats), len(lngs), max_len)
            for j in range(seq_len):
                gps_data[i, j, 0] = lats[j]  # 纬度
                gps_data[i, j, 1] = lngs[j]  # 经度
    
    # 构造测试数据元组
    test_data = (route_data, None, gps_data, None, None,
                 grid_data, None, gps_data_grid, None, None,
                 None, None, test_seq_data)
    
    # 提取序列嵌入
    st = time.time()
    seq_embedding = get_seq_emb_from_traj_withALLModel(seq_model, test_data, batch_size=64)
    print(f"嵌入提取耗时：{time.time() - st:.2f}s")
    
    # 进行时间估计评估
    evaluation_with_fused_representation(seq_embedding, test_seq_data, road_network_size)

    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))


if __name__ == '__main__':
    # 设置参数
    city = 'chengdu'
    exp_path = './checkpoints'
    model_name = 'best_model.pt'
    
    start_time = time.time()
    evaluation(city, exp_path, model_name, start_time)