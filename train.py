#!/usr/bin/python
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import random
import logging
import time
from tqdm import tqdm
from datetime import datetime
import json

from model.TrajFuse import TrajFuse  # 使用修复后的模型
from dataloader import get_trifusion_loaders

from metrics_simple import cal_pre_recall_with_f1

'''
train:
    python train.py
test:
    python train.py --test-only --resume=/home/sjj/liyantao/TrajFuse/checkpoints/best_model_1e4.pt

python /home/sjj/liyantao/TrajFuse_代码/train.py --test-only --resume=/home/sjj/liyantao/TrajFuse/checkpoints/best_model_1e4.pt
'''

# 设置日志
def setup_logging(log_dir):
    """设置日志记录"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('TriFusion')

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_strict_deterministic(seed, enable_strict=False):
    """设置严格确定性模式，包括单线程和严格CUDA确定性"""
    import os
    
    # 首先设置基础随机种子
    set_seed(seed)
    
    if enable_strict:
        print("启用严格确定性模式...")
        print("   - 单线程执行")
        print("   - 严格CUDA确定性")
        print("   - 可能会影响训练性能")
        
        # 强制单线程模式
        torch.set_num_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # 更严格的CUDA确定性
        if torch.cuda.is_available():
            # 使用PyTorch 1.8+的确定性算法
            try:
                torch.use_deterministic_algorithms(True)
                print("启用 torch.use_deterministic_algorithms")
            except Exception as e:
                print(f" torch.use_deterministic_algorithms 不可用: {e}")
            
            # 确保CUDA操作的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # 设置CUDA全局标志
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            
        print("严格确定性模式已启用")
    else:
        # 在 set_seed 函数后添加  保证了结果复现
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        print("使用标准确定性模式（允许并行计算以提高性能）")

def validate_tensor(tensor, name, logger):
    """验证张量的数值稳定性"""
    if tensor is None:
        return True
    
    if torch.isnan(tensor).any():
        logger.error(f"{name} 包含 NaN 值")
        return False
    
    if torch.isinf(tensor).any():
        logger.error(f"{name} 包含无穷大值")
        return False
    
    if torch.abs(tensor).max() > 1e6:
        logger.warning(f"{name} 包含极大值: {torch.abs(tensor).max().item()}")
        return False
    
    return True

def safe_backward_and_step(loss, model, optimizer, grad_clip, logger):
    """安全的反向传播和参数更新"""
    try:
        # 检查损失有效性
        if not validate_tensor(loss, "loss", logger):
            logger.error("损失无效，跳过当前批次")
            return False
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        total_norm = 0
        param_count = 0
        nan_grad_count = 0
        


        for name, param in model.named_parameters():
            if param.grad is not None:
                param_count += 1


                
                # 检查梯度中的NaN
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    if torch.isnan(param.grad).any():
                        print(f"参数 {name} 的梯度包含NaN值")
                        #logger.warning(f"参数 {name} 的梯度包含NaN值")
                    if torch.isinf(param.grad).any():
                        logger.warning(f"参数 {name} 的梯度包含无穷大")
                    nan_grad_count += 1
                    # 将NaN梯度置零
                    param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 计算梯度范数
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        
        
        # 如果有太多NaN梯度，跳过更新
        if nan_grad_count > param_count * 0.1:  # 超过10%的参数有NaN梯度
            logger.error(f"过多参数({nan_grad_count}/{param_count})存在NaN梯度，跳过更新")
            optimizer.zero_grad()
            return False
        
        #print(f"总梯度范数: {total_norm:.10f}")
        
        # 梯度裁剪
        if grad_clip is not None and total_norm > grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            logger.debug(f"梯度被裁剪: {total_norm:.4f} -> {grad_clip}")
        
        # 参数更新
        optimizer.step()
        
        # 检查更新后的参数
        nan_param_count = 0
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                logger.error(f"参数 {name} 在更新后包含NaN或无穷大")
                nan_param_count += 1
        
        if nan_param_count > 0:
            logger.error(f"有 {nan_param_count} 个参数在更新后异常")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"反向传播过程出错: {e}")
        return False

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer=None, grad_clip=None, logger=None, enable_modal_dropout=False, modal_dropout_prob=0.8):
    """训练一个epoch - 数值稳定版本，支持新的模型架构，支持模态数据随机丢弃"""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_mlm_loss = 0
    total_sequence_reg = 0
    total_samples = 0
    total_contrastive_loss = 0  #对比学习损失
    total_contrastive_temperature = 0  # 对比学习温度参数
    successful_batches = 0
    failed_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        try:
            # 提取基本数据
            gps_data = batch['gps_data'].to(device) if batch['gps_data'] is not None else None
            cdr_data = batch['cdr_data'].to(device) if batch['cdr_data'] is not None else None
            road_seg_data = batch['road_gps_data'].to(device) if batch['road_gps_data'] is not None else None
            target_seg_data = batch['target_seg_data'].to(device)
            target_seg_mask = batch['target_seg_mask'].to(device)
            road_network = batch['road_network'].to(device)
            
            # 模态数据随机丢弃
            if enable_modal_dropout:
                batch_size = target_seg_data.size(0)
                for sample_idx in range(batch_size):
                    # 随机生成一个概率
                    drop_prob = random.random()
                    if drop_prob > modal_dropout_prob:
                        # 确定可以丢弃的模态列表（只考虑非None的模态）
                        available_modalities = []
                        if gps_data is not None:
                            available_modalities.append('gps')
                        if cdr_data is not None:
                            available_modalities.append('cdr')
                        if road_seg_data is not None:
                            available_modalities.append('road_seg')
                        
                        # 如果有可用的模态数据，随机选择一种丢弃
                        if len(available_modalities) > 1:  # 至少保留一种模态
                            modality_to_drop = random.choice(available_modalities)
                            
                            if modality_to_drop == 'gps' and gps_data is not None:
                                # 将该样本的GPS数据置零
                                gps_data[sample_idx] = torch.zeros_like(gps_data[sample_idx])
                            elif modality_to_drop == 'cdr' and cdr_data is not None:
                                # 将该样本的CDR数据置零
                                cdr_data[sample_idx] = torch.zeros_like(cdr_data[sample_idx])
                            elif modality_to_drop == 'road_seg' and road_seg_data is not None:
                                # 将该样本的road_seg数据置零
                                road_seg_data[sample_idx] = torch.zeros_like(road_seg_data[sample_idx])
            
            # 提取扩展数据字段
            kwargs = {}
            if 'sparse_gps2route_data' in batch and batch['sparse_gps2route_data'] is not None:
                kwargs['sparse_gps2route_data'] = batch['sparse_gps2route_data'].to(device)
            if 'cdr_gps2route_data' in batch and batch['cdr_gps2route_data'] is not None:
                kwargs['cdr_gps2route_data'] = batch['cdr_gps2route_data'].to(device)
            if 'monitor_opath_data' in batch and batch['monitor_opath_data'] is not None:
                kwargs['monitor_opath_data'] = batch['monitor_opath_data'].to(device)
            if 'merged_route_data' in batch and batch['merged_route_data'] is not None:
                kwargs['merged_route_data'] = batch['merged_route_data'].to(device)
            if 'original_cpath_data' in batch and batch['original_cpath_data'] is not None:
                kwargs['original_cpath_data'] = batch['original_cpath_data'].to(device)
            else:
                # 如果没有original_cpath_data，使用target_seg_data
                kwargs['original_cpath_data'] = target_seg_data

            # 提取扩展时间字段
            if 'gps_time_data' in batch and batch['gps_time_data'] is not None:
                kwargs['gps_time_data'] = batch['gps_time_data'].to(device)
            if 'cdr_time_data' in batch and batch['cdr_time_data'] is not None:
                kwargs['cdr_time_data'] = batch['cdr_time_data'].to(device)
            if 'road_gps_time_data' in batch and batch['road_gps_time_data'] is not None:
                kwargs['road_time_data'] = batch['road_gps_time_data'].to(device)
            
            # 数据有效性检查
            if not validate_tensor(gps_data, "gps_data", logger):
                continue
            if not validate_tensor(cdr_data, "cdr_data", logger):
                continue
            if not validate_tensor(target_seg_data, "target_seg_data", logger):
                continue
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            pred_segments, loss_dict, _ = model(
                gps_data=gps_data,
                cdr_data=cdr_data,
                road_seg_data=road_seg_data,
                road_network=road_network,
                target=target_seg_data,
                **kwargs
            )
            
            # 获取损失
            loss = loss_dict['total_loss']
            
            # 安全的反向传播和更新
            if safe_backward_and_step(loss, model, optimizer, grad_clip, logger):
                # 统计成功的批次
                batch_size = target_seg_data.size(0)
                total_loss += loss.item() * batch_size
                total_main_loss += loss_dict.get('main_loss', 0) * batch_size
                total_mlm_loss += loss_dict.get('mlm_loss', 0) * batch_size
                total_contrastive_loss += loss_dict.get('contrastive_loss', 0) * batch_size  #对比学习损失
                total_contrastive_temperature += loss_dict.get('contrastive_temperature', 0) * batch_size  # 对比学习温度参数
                total_sequence_reg += loss_dict.get('sequence_reg', 0) * batch_size
                total_samples += batch_size
                successful_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'main_loss': f"{loss_dict.get('main_loss', 0):.4f}",
                    'mlm_loss': f"{loss_dict.get('mlm_loss', 0):.4f}",
                    'con_loss': f"{loss_dict.get('contrastive_loss', 0):.4f}",  #对比学习损失
                    'con_temp': f"{loss_dict.get('contrastive_temperature', 0):.4f}",  # 对比学习温度参数
                    'seq_reg': f"{loss_dict.get('sequence_reg', 0):.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                    'success_rate': f"{successful_batches/(successful_batches+failed_batches):.2f}"
                })
                
                # 记录到TensorBoard
                if writer is not None and batch_idx % 10 == 0:
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('train/loss', loss.item(), global_step)
                    writer.add_scalar('train/main_loss', loss_dict.get('main_loss', 0), global_step)
                    writer.add_scalar('train/mlm_loss', loss_dict.get('mlm_loss', 0), global_step)
                    writer.add_scalar('train/contrastive_loss', loss_dict.get('contrastive_loss', 0), global_step)  #new 对比学习损失
                    writer.add_scalar('train/contrastive_temperature', loss_dict.get('contrastive_temperature', 0), global_step)  # 对比学习温度参数
                    writer.add_scalar('train/sequence_reg', loss_dict.get('sequence_reg', 0), global_step)
                    
                    
            else:
                failed_batches += 1
                logger.warning(f"批次 {batch_idx} 训练失败")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"批次 {batch_idx} GPU内存不足: {e}")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.error(f"批次 {batch_idx} 运行时错误: {e}")

                
            failed_batches += 1
            continue
            
        except Exception as e:
            logger.error(f"批次 {batch_idx} 发生未知错误: {e}")
            
            failed_batches += 1
            continue
    
    # 计算平均损失
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_main_loss = total_main_loss / total_samples
        avg_mlm_loss = total_mlm_loss / total_samples
        avg_contrastive_loss = total_contrastive_loss / total_samples  # 对比学习损失
        avg_contrastive_temperature = total_contrastive_temperature / total_samples  # 对比学习温度参数
        avg_sequence_reg = total_sequence_reg / total_samples
    else:
        avg_loss = float('inf')
        avg_main_loss = float('inf')
        avg_mlm_loss = float('inf')
        avg_contrastive_loss = float('inf')  # 对比学习损失
        avg_contrastive_temperature = float('inf')  # 对比学习温度参数
        avg_sequence_reg = float('inf')
    
    # 记录成功率
    total_batches = successful_batches + failed_batches
    success_rate = successful_batches / total_batches if total_batches > 0 else 0
    
    logger.info(f"Epoch {epoch} 训练完成: 成功批次 {successful_batches}/{total_batches} ({success_rate:.2%})")
    
    return {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'mlm_loss': avg_mlm_loss,
        'contrastive_loss': avg_contrastive_loss,  #new 对比学习损失
        'contrastive_temperature': avg_contrastive_temperature,  # 对比学习温度参数
        'sequence_reg': avg_sequence_reg,
        'success_rate': success_rate
    }

def validate(model, val_loader, criterion, device, logger):
    """验证模型性能 - 数值稳定版本，支持新的模型架构"""
    model.eval()
    total_loss = 0
    total_main_loss = 0
    total_mlm_loss = 0
    total_contrastive_loss = 0  #new 对比学习损失
    total_contrastive_temperature = 0  # 对比学习温度参数
    total_sequence_reg = 0
    total_samples = 0
    successful_batches = 0
    failed_batches = 0
    
    # 用于计算指标
    all_preds_list = []
    all_targets_list = []
    #all_intermediate_preds_list = []  # 新增：收集intermediate generator的预测结果
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
            try:
                # 提取基本数据
                gps_data = batch['gps_data'].to(device) if batch['gps_data'] is not None else None
                cdr_data = batch['cdr_data'].to(device) if batch['cdr_data'] is not None else None
                road_seg_data = batch['road_seg_data'].to(device) if batch['road_seg_data'] is not None else None
                road_seg_data = batch['road_gps_data'].to(device) if batch['road_gps_data'] is not None else None

                target_seg_data = batch['target_seg_data'].to(device)
                road_network = batch['road_network'].to(device)
                
                # 提取扩展数据字段
                kwargs = {}
                if 'sparse_gps2route_data' in batch and batch['sparse_gps2route_data'] is not None:
                    kwargs['sparse_gps2route_data'] = batch['sparse_gps2route_data'].to(device)
                if 'cdr_gps2route_data' in batch and batch['cdr_gps2route_data'] is not None:
                    kwargs['cdr_gps2route_data'] = batch['cdr_gps2route_data'].to(device)
                if 'monitor_opath_data' in batch and batch['monitor_opath_data'] is not None:
                    kwargs['monitor_opath_data'] = batch['monitor_opath_data'].to(device)
                if 'merged_route_data' in batch and batch['merged_route_data'] is not None:
                    kwargs['merged_route_data'] = batch['merged_route_data'].to(device)
                if 'original_cpath_data' in batch and batch['original_cpath_data'] is not None:
                    kwargs['original_cpath_data'] = batch['original_cpath_data'].to(device)
                else:
                    kwargs['original_cpath_data'] = target_seg_data

                # 提取扩展时间字段
                if 'gps_time_data' in batch and batch['gps_time_data'] is not None:
                    kwargs['gps_time_data'] = batch['gps_time_data'].to(device)
                if 'cdr_time_data' in batch and batch['cdr_time_data'] is not None:
                    kwargs['cdr_time_data'] = batch['cdr_time_data'].to(device)
                #if 'road_time_data' in batch and batch['road_time_data'] is not None:
                #    kwargs['road_time_data'] = batch['road_time_data'].to(device)
                if 'road_gps_time_data' in batch and batch['road_gps_time_data'] is not None:
                    kwargs['road_time_data'] = batch['road_gps_time_data'].to(device)
                
                # 数据有效性检查
                if not validate_tensor(target_seg_data, "target_seg_data", logger):
                    failed_batches += 1
                    continue
                
                # 检查目标数据范围
                road_network_size = road_network.x.size(0)
                max_id = target_seg_data.max().item()
                if max_id >= road_network_size:
                    logger.warning(f"批次 {batch_idx} 目标ID超出范围: {max_id} >= {road_network_size}")
                    failed_batches += 1
                    continue
                
                # 前向传播
                pred_segments, loss_dict, _ = model(
                    gps_data=gps_data,
                    cdr_data=cdr_data,
                    road_seg_data=road_seg_data,
                    road_network=road_network,
                    target=target_seg_data,
                    **kwargs
                )
                
                # 检查输出有效性
                if not validate_tensor(pred_segments, "pred_segments", logger):
                    failed_batches += 1
                    continue
                
                if not validate_tensor(loss_dict['total_loss'], "total_loss", logger):
                    failed_batches += 1
                    continue
                
                # 记录损失
                batch_size = target_seg_data.size(0)
                total_loss += loss_dict['total_loss'].item() * batch_size
                total_main_loss += loss_dict.get('main_loss', 0) * batch_size
                total_mlm_loss += loss_dict.get('mlm_loss', 0) * batch_size
                total_contrastive_loss += loss_dict.get('contrastive_loss', 0) * batch_size  #new 对比学习损失
                total_contrastive_temperature += loss_dict.get('contrastive_temperature', 0) * batch_size  # 对比学习温度参数
                total_sequence_reg += loss_dict.get('sequence_reg', 0) * batch_size
                total_samples += batch_size
                successful_batches += 1
                
                # 计算预测结果
                _, pred_indices = torch.max(pred_segments, dim=-1)
                
                # 检查预测的有效性
                if pred_indices.max() >= road_network_size:
                    logger.warning(f"批次 {batch_idx} 预测ID超出范围，进行裁剪")
                    pred_indices = torch.clamp(pred_indices, 0, road_network_size-1)
                
                # 收集预测和目标
                all_preds_list.extend(pred_indices.cpu().numpy().tolist())
                all_targets_list.extend(target_seg_data.cpu().numpy().tolist())
                
                
            except Exception as e:
                logger.error(f"验证批次 {batch_idx} 发生错误: {e}")
                failed_batches += 1
                continue
    
    # 计算平均损失
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_main_loss = total_main_loss / total_samples
        avg_mlm_loss = total_mlm_loss / total_samples
        avg_contrastive_loss = total_contrastive_loss / total_samples  #new 对比学习损失
        avg_contrastive_temperature = total_contrastive_temperature / total_samples  # 对比学习温度参数
        avg_sequence_reg = total_sequence_reg / total_samples
    else:
        avg_loss = float('inf')
        avg_main_loss = float('inf')
        avg_mlm_loss = float('inf')
        avg_contrastive_loss = float('inf')  #new 对比学习损失
        avg_contrastive_temperature = float('inf')  # 对比学习温度参数
        avg_sequence_reg = float('inf')
        logger.error("验证集没有成功处理的批次")
    
    # 初始化指标字典
    metrics_results = {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'mlm_loss': avg_mlm_loss,
        'contrastive_loss': avg_contrastive_loss,  #new 对比学习损失
        'contrastive_temperature': avg_contrastive_temperature,  # 对比学习温度参数
        'sequence_reg': avg_sequence_reg,
        'rid_acc': 0.0, 
        'rid_recall': 0.0, 
        'rid_precision': 0.0, 
        'rid_f1': 0.0,
        'success_rate': successful_batches / (successful_batches + failed_batches) if (successful_batches + failed_batches) > 0 else 0
    }

    # 计算预测指标
    if all_preds_list and all_targets_list and successful_batches > 0:
        try:
            detailed_metrics = cal_pre_recall_with_f1(all_preds_list, all_targets_list, eos_token_id=-1)
            metrics_results.update(detailed_metrics)
            # 为了向后兼容，添加一些别名
            metrics_results['lcs_recall'] = detailed_metrics.get('rid_recall', 0.0)
            metrics_results['lcs_precision'] = detailed_metrics.get('rid_precision', 0.0)
            metrics_results['lcs_f1'] = detailed_metrics.get('rid_f1', 0.0)
            metrics_results['trad_acc'] = detailed_metrics.get('rid_acc', 0.0)
        except Exception as e:
            logger.error(f"指标计算失败: {e}")
    

    return metrics_results




def calculate_static_input_target_metrics(val_loader, device, logger, eos_token_id=-1):
    """计算静态输入-目标指标 - 数值稳定版本"""
    logger.info("Calculating static input-target metrics...")
    all_input_road_seg_list = []
    all_targets_for_input_metrics_list = []
    successful_batches = 0
    failed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Static Input-Target Metrics Calc')):
            try:
                road_seg_data = batch['road_seg_data'].to(device) if batch['road_seg_data'] is not None else None
                target_seg_data = batch['target_seg_data'].to(device)

                if road_seg_data is not None and target_seg_data is not None:
                    # 数据有效性检查
                    if not validate_tensor(road_seg_data, "road_seg_data", logger):
                        failed_batches += 1
                        continue
                    if not validate_tensor(target_seg_data, "target_seg_data", logger):
                        failed_batches += 1
                        continue
                    
                    current_input_road_segs = road_seg_data.cpu().numpy().tolist()
                    current_targets_for_input = target_seg_data.cpu().numpy().tolist()
                    
                    if len(current_input_road_segs) == len(current_targets_for_input):
                        all_input_road_seg_list.extend(current_input_road_segs)
                        all_targets_for_input_metrics_list.extend(current_targets_for_input)
                        successful_batches += 1
                    else:
                        logger.warning(f"静态指标计算: 批次 {batch_idx} 输入输出长度不匹配")
                        failed_batches += 1
                elif road_seg_data is None:
                    logger.debug(f"静态指标计算: 批次 {batch_idx} 没有road_seg_data")
                    failed_batches += 1

            except Exception as e:
                logger.error(f"静态指标计算: 批次 {batch_idx} 错误: {e}")
                failed_batches += 1
                continue
    
    static_metrics = {}
    default_keys = ['rid_acc', 'rid_recall', 'rid_precision', 'rid_f1']

    if all_input_road_seg_list and all_targets_for_input_metrics_list and successful_batches > 0:
        try:
            logger.info(f"计算静态输入-目标指标 ({len(all_input_road_seg_list)} 样本，成功率: {successful_batches/(successful_batches+failed_batches):.2%})")
            calculated_metrics = cal_pre_recall_with_f1(
                all_input_road_seg_list, 
                all_targets_for_input_metrics_list, 
                eos_token_id=eos_token_id
            )
            for k, v in calculated_metrics.items():
                static_metrics[f'static_it_{k}'] = v
            
            # 为了向后兼容，添加一些别名
            static_metrics['static_it_lcs_recall'] = calculated_metrics.get('rid_recall', 0.0)
            static_metrics['static_it_lcs_precision'] = calculated_metrics.get('rid_precision', 0.0)
            static_metrics['static_it_lcs_f1'] = calculated_metrics.get('rid_f1', 0.0)
            static_metrics['static_it_trad_acc'] = calculated_metrics.get('rid_acc', 0.0)
            
            log_it_metrics = {f'static_it_{k}': static_metrics.get(f'static_it_{k}', 0.0) for k in default_keys}
            logger.info(f"静态输入-目标指标: {log_it_metrics}")
        except Exception as e:
            logger.error(f"静态指标计算失败: {e}")
            for k in default_keys:
                static_metrics[f'static_it_{k}'] = 0.0
    else:
        logger.warning("静态输入-目标指标计算数据不足")
        for k in default_keys:
            static_metrics[f'static_it_{k}'] = 0.0
            
    return static_metrics

def calculate_model_generation_metrics(model, data_loader, device, logger, eos_token_id=-1):
    """
    计算模型生成指标 - 使用模型的final_generator生成序列并计算指标
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备
        logger: 日志记录器
        eos_token_id: 结束token ID
        
    Returns:
        metrics_dict: 包含rid_acc, rid_recall, rid_precision, rid_f1等指标的字典
    """
    model.eval()
    all_generated_sequences = []
    all_target_sequences = []
    successful_batches = 0
    failed_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Model Generation Metrics Calc')):
            try:
                # 提取基本数据
                gps_data = batch['gps_data'].to(device) if batch['gps_data'] is not None else None
                cdr_data = batch['cdr_data'].to(device) if batch['cdr_data'] is not None else None
                road_seg_data = batch['road_seg_data'].to(device) if batch['road_seg_data'] is not None else None
                road_seg_data = batch['road_gps_data'].to(device) if batch['road_gps_data'] is not None else None
                

                target_seg_data = batch['target_seg_data'].to(device)
                road_network = batch['road_network'].to(device)
                
                # 提取扩展数据字段
                kwargs = {}
                if 'sparse_gps2route_data' in batch and batch['sparse_gps2route_data'] is not None:
                    kwargs['sparse_gps2route_data'] = batch['sparse_gps2route_data'].to(device)
                if 'cdr_gps2route_data' in batch and batch['cdr_gps2route_data'] is not None:
                    kwargs['cdr_gps2route_data'] = batch['cdr_gps2route_data'].to(device)
                if 'monitor_opath_data' in batch and batch['monitor_opath_data'] is not None:
                    kwargs['monitor_opath_data'] = batch['monitor_opath_data'].to(device)
                if 'merged_route_data' in batch and batch['merged_route_data'] is not None:
                    kwargs['merged_route_data'] = batch['merged_route_data'].to(device)
                if 'original_cpath_data' in batch and batch['original_cpath_data'] is not None:
                    kwargs['original_cpath_data'] = batch['original_cpath_data'].to(device)
                else:
                    kwargs['original_cpath_data'] = target_seg_data

                # 提取扩展时间字段
                if 'gps_time_data' in batch and batch['gps_time_data'] is not None:
                    kwargs['gps_time_data'] = batch['gps_time_data'].to(device)
                if 'cdr_time_data' in batch and batch['cdr_time_data'] is not None:
                    kwargs['cdr_time_data'] = batch['cdr_time_data'].to(device)
                #if 'road_time_data' in batch and batch['road_time_data'] is not None:
                #    kwargs['road_time_data'] = batch['road_time_data'].to(device)
                if 'road_gps_time_data' in batch and batch['road_gps_time_data'] is not None:
                    kwargs['road_time_data'] = batch['road_gps_time_data'].to(device)

                # 数据有效性检查
                if not validate_tensor(target_seg_data, "target_seg_data", logger):
                    failed_batches += 1
                    continue
                
                # 检查目标数据范围
                road_network_size = road_network.x.size(0)
                max_id = target_seg_data.max().item()
                if max_id >= road_network_size:
                    logger.warning(f"生成指标批次 {batch_idx} 目标ID超出范围: {max_id} >= {road_network_size}")
                    failed_batches += 1
                    continue
                
                # 模型前向传播生成序列
                pred_segments, _, _ = model(
                    gps_data=gps_data,
                    cdr_data=cdr_data,
                    road_seg_data=road_seg_data,
                    road_network=road_network,
                    target=None,  # 推理模式，不提供target
                    **kwargs
                )
                
                # 检查输出有效性
                if not validate_tensor(pred_segments, "pred_segments", logger):
                    failed_batches += 1
                    continue
                
                # 计算预测结果
                _, pred_indices = torch.max(pred_segments, dim=-1)
                
                # 检查预测的有效性
                if pred_indices.max() >= road_network_size:
                    logger.warning(f"生成指标批次 {batch_idx} 预测ID超出范围，进行裁剪")
                    pred_indices = torch.clamp(pred_indices, 0, road_network_size-1)

                # 收集生成序列和目标序列
                all_generated_sequences.extend(pred_indices.cpu().numpy().tolist())
                all_target_sequences.extend(target_seg_data.cpu().numpy().tolist())
                successful_batches += 1
                
            except Exception as e:
                logger.error(f"生成指标计算批次 {batch_idx} 发生错误: {e}")
                failed_batches += 1
                exit(1)
                continue
    
    # 初始化指标字典
    generation_metrics = {
        'rid_acc': 0.0, 
        'rid_recall': 0.0, 
        'rid_precision': 0.0, 
        'rid_f1': 0.0,
        'success_rate': successful_batches / (successful_batches + failed_batches) if (successful_batches + failed_batches) > 0 else 0
    }
    
    # 计算指标
    if all_generated_sequences and all_target_sequences and successful_batches > 0:
        try:
            logger.info(f"计算模型生成指标 ({len(all_generated_sequences)} 样本，成功率: {generation_metrics['success_rate']:.2%})")
            detailed_metrics = cal_pre_recall_with_f1(all_generated_sequences, all_target_sequences, eos_token_id=eos_token_id)
            generation_metrics.update(detailed_metrics)
            # 为了向后兼容，添加一些别名
            generation_metrics['lcs_recall'] = detailed_metrics.get('rid_recall', 0.0)
            generation_metrics['lcs_precision'] = detailed_metrics.get('rid_precision', 0.0)
            generation_metrics['lcs_f1'] = detailed_metrics.get('rid_f1', 0.0)
            generation_metrics['trad_acc'] = detailed_metrics.get('rid_acc', 0.0)
        except Exception as e:
            logger.error(f"生成指标计算失败: {e}")
    else:
        logger.warning("模型生成指标计算数据不足")
    
    return generation_metrics



def test(model, test_loader, device, logger, save_dir=None, eos_token_id=-1):
    """测试模型 - 数值稳定版本，支持新的模型架构"""
    model.eval()
    
    all_preds_list = []
    all_targets_list = []
    successful_batches = 0
    failed_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            try:
                # 提取基本数据
                gps_data = batch['gps_data'].to(device) if batch['gps_data'] is not None else None
                cdr_data = batch['cdr_data'].to(device) if batch['cdr_data'] is not None else None
                road_seg_data = batch['road_gps_data'].to(device) if batch['road_gps_data'] is not None else None
                target_seg_data = batch['target_seg_data'].to(device)
                road_network = batch['road_network'].to(device)
                
                # 提取扩展数据字段
                kwargs = {}
                if 'sparse_gps2route_data' in batch and batch['sparse_gps2route_data'] is not None:
                    kwargs['sparse_gps2route_data'] = batch['sparse_gps2route_data'].to(device)
                if 'cdr_gps2route_data' in batch and batch['cdr_gps2route_data'] is not None:
                    kwargs['cdr_gps2route_data'] = batch['cdr_gps2route_data'].to(device)
                if 'monitor_opath_data' in batch and batch['monitor_opath_data'] is not None:
                    kwargs['monitor_opath_data'] = batch['monitor_opath_data'].to(device)
                if 'merged_route_data' in batch and batch['merged_route_data'] is not None:
                    kwargs['merged_route_data'] = batch['merged_route_data'].to(device)
                if 'original_cpath_data' in batch and batch['original_cpath_data'] is not None:
                    kwargs['original_cpath_data'] = batch['original_cpath_data'].to(device)
                else:
                    kwargs['original_cpath_data'] = target_seg_data
                
                # 提取扩展时间字段
                if 'gps_time_data' in batch and batch['gps_time_data'] is not None:
                    kwargs['gps_time_data'] = batch['gps_time_data'].to(device)
                if 'cdr_time_data' in batch and batch['cdr_time_data'] is not None:
                    kwargs['cdr_time_data'] = batch['cdr_time_data'].to(device)
                if 'road_gps_time_data' in batch and batch['road_gps_time_data'] is not None:
                    kwargs['road_time_data'] = batch['road_gps_time_data'].to(device)

                # 数据有效性检查
                if not validate_tensor(target_seg_data, "target_seg_data", logger):
                    failed_batches += 1
                    continue
                
                # 前向传播
                pred_segments, _, intermediate_pred = model(
                    gps_data=gps_data,
                    cdr_data=cdr_data,
                    road_seg_data=road_seg_data,
                    road_network=road_network,
                    target=target_seg_data,
                    **kwargs
                )
                
                # 检查预测有效性
                if not validate_tensor(pred_segments, "pred_segments", logger):
                    failed_batches += 1
                    continue
                
                # 计算预测结果
                _, pred_indices = torch.max(pred_segments, dim=-1)

                # 检查预测范围
                road_network_size = road_network.x.size(0)
                if pred_indices.max() >= road_network_size:
                    logger.warning(f"测试批次 {batch_idx} 预测ID超出范围，进行裁剪")
                    pred_indices = torch.clamp(pred_indices, 0, road_network_size-1)
                
                # 收集预测和目标
                all_preds_list.extend(pred_indices.cpu().numpy().tolist())
                all_targets_list.extend(target_seg_data.cpu().numpy().tolist())


                successful_batches += 1
                
            except Exception as e:
                logger.error(f"测试批次 {batch_idx} 发生错误: {e}")
                failed_batches += 1
                continue
    
    # 初始化指标字典
    final_metrics = {
        'rid_acc': 0.0, 
        'rid_recall': 0.0, 
        'rid_precision': 0.0, 
        'rid_f1': 0.0,
        'success_rate': successful_batches / (successful_batches + failed_batches) if (successful_batches + failed_batches) > 0 else 0
    }

    # 计算指标
    if all_preds_list and all_targets_list and successful_batches > 0:
        try:
            detailed_metrics = cal_pre_recall_with_f1(all_preds_list, all_targets_list, eos_token_id=eos_token_id)
            final_metrics.update(detailed_metrics)
            # 为了向后兼容，添加一些别名
            final_metrics['lcs_recall'] = detailed_metrics.get('rid_recall', 0.0)
            final_metrics['lcs_precision'] = detailed_metrics.get('rid_precision', 0.0)
            final_metrics['lcs_f1'] = detailed_metrics.get('rid_f1', 0.0)
            final_metrics['trad_acc'] = detailed_metrics.get('rid_acc', 0.0)
        except Exception as e:
            logger.error(f"测试指标计算失败: {e}")
    else:
        logger.warning("测试集没有成功处理的批次")
    


    # 记录测试结果
    log_message = f"Test Results (成功率: {final_metrics['success_rate']:.2%}): "
    for k, v in final_metrics.items():
        if k != 'success_rate':
            log_message += f"{k}: {v:.4f} "
    logger.info(log_message)
    
    # 保存结果
    if save_dir:
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            result_file = os.path.join(save_dir, 'test_results.pt')
            torch.save({
                'metrics': final_metrics,
                'successful_batches': successful_batches,
                'failed_batches': failed_batches
            }, result_file)
            
            logger.info(f'测试结果保存至 {result_file}')
        except Exception as e:
            logger.error(f"保存测试结果失败: {e}")
    
    return final_metrics








def main():
    """主函数 - 数值稳定版本"""
    parser = argparse.ArgumentParser(description='Train TriFusion-PathNet model (数值稳定版)')
    
    
    # 数据参数
    parser.add_argument('--data-path', type=str, default="/root/autodl-tmp/data/chengdu/chengdu_train_extension65_50_35_road2gps.pkl", help='Path to the data file')
    parser.add_argument('--route-min-len', type=int, default=5, help='Minimum length of route segments')
    parser.add_argument('--route-max-len', type=int, default=256, help='Maximum length of route segments100')
    parser.add_argument('--gps-min-len', type=int, default=10, help='Minimum length of GPS trajectory')
    parser.add_argument('--gps-max-len', type=int, default=256, help='Maximum length of GPS trajectory')
    parser.add_argument('--cdr-min-len', type=int, default=2, help='Minimum length of CDR trajectory')
    parser.add_argument('--cdr-max-len', type=int, default=256, help='Maximum length of CDR trajectory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (reduced for stability)32')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--num-samples', type=int, default=98000, help='Number of samples to use，98000')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Ratio of test data')
    
    # 模型参数 (调整为更保守的设置)
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads (reduced)')
    parser.add_argument('--num-encoder-layers', type=int, default=3, help='Number of encoder layers (reduced)')
    parser.add_argument('--num-decoder-layers', type=int, default=3, help='Number of decoder layers (reduced)')
    parser.add_argument('--dim-feedforward', type=int, default=512, help='Dimension of feedforward networks (reduced)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--mlm-loss-weight', type=float, default=0.5, help='MLM任务损失权重')
    parser.add_argument('--mlm-mask-prob', type=float, default=0.15, help='MLM掩码概率')
    parser.add_argument('--mlm-replace-prob', type=float, default=0.8, help='MLM替换概率')
    parser.add_argument('--mlm-random-prob', type=float, default=0.1, help='MLM随机替换概率')
    parser.add_argument('--main-loss-weight', type=float, default=1.0, help='Weight for main task loss')
    parser.add_argument('--contrastive-loss-weight', type=float, default=0.5, help='Weight for contrastive loss 0.5')  #new 对比学习损失权重
    parser.add_argument('--contrastive-temperature', type=float, default=0.1, help='Temperature for contrastive loss 0.1')  #new 对比学习温度参数



    parser.add_argument('--sequence-reg-weight', type=float, default=0.1, help='Weight for sequence regularization loss')
    
    # 训练参数 (更保守的设置)
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (reduced)0.001')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay1e-5/6')
    parser.add_argument('--modal-dropout-prob', type=float, default=1.0, help='Modal dropout probability threshold (当随机数>此值时丢弃一种模态，默认0.8)')
    parser.add_argument('--enable-modal-dropout', action='store_true', help='启用训练时模态数据随机丢弃功能')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value (reduced)梯度裁剪值，默认是1.0')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    parser.add_argument('--strict-deterministic', action='store_true', 
                        help='启用严格确定性模式：单线程 + 严格CUDA确定性 (可能影响性能但确保完全一致的结果)')
    parser.add_argument('--save-dir', type=str, default='/root/TrajFuse_代码/checkpoints', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='/root/TrajFuse_代码/logs', help='Directory to save logs')
    parser.add_argument('--resume', type=str, default='/root/TrajFuse/checkpoints/best_model_1e4.pt', help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true', help='Only run testing')
    parser.add_argument('--eval-only', action='store_true', help='Only load model and test eval data loading (for debugging eval data issues)')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity of logging')
    parser.add_argument('--debug-cuda', action='store_true', help='Enable CUDA_LAUNCH_BLOCKING=1 for better error reporting')
    
    args = parser.parse_args()
     
    
    # 设置随机种子和确定性模式
    set_strict_deterministic(args.seed, enable_strict=args.strict_deterministic)
    
    # 设置日志
    logger = setup_logging(args.log_dir)
    
    # 设置日志级别
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # 设置CUDA断言调试
    if args.debug_cuda:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        logger.info("已启用CUDA_LAUNCH_BLOCKING=1进行调试")
    
    # 详细记录所有参数配置
    logger.info("=" * 80)
    logger.info("训练参数配置详情")
    logger.info("=" * 80)
    
    # 数据参数
    logger.info("【数据参数】")
    logger.info(f"  数据路径: {args.data_path}")
    logger.info(f"  批次大小: {args.batch_size}")
    logger.info(f"  工作线程数: {args.num_workers}")
    logger.info(f"  样本数量: {args.num_samples}")
    logger.info(f"  测试集比例: {args.test_ratio}")
    logger.info(f"  Road轨迹长度范围: {args.route_min_len} ~ {args.route_max_len}")
    logger.info(f"  GPS轨迹长度范围: {args.gps_min_len} ~ {args.gps_max_len}")
    logger.info(f"  CDR轨迹长度范围: {args.cdr_min_len} ~ {args.cdr_max_len}")
    
    # 模型参数
    logger.info("【模型参数】")
    logger.info(f"  模型维度 (d_model): {args.d_model}")
    logger.info(f"  注意力头数 (nhead): {args.nhead}")
    logger.info(f"  编码器层数: {args.num_encoder_layers}")
    logger.info(f"  解码器层数: {args.num_decoder_layers}")
    logger.info(f"  前馈网络维度: {args.dim_feedforward}")
    logger.info(f"  Dropout率: {args.dropout}")
    
    # 损失权重参数
    logger.info("【损失权重参数】")
    logger.info(f"  主任务损失权重: {args.main_loss_weight}")
    logger.info(f"  对比学习损失权重: {args.contrastive_loss_weight}")
    logger.info(f"  对比学习温度参数: {args.contrastive_temperature}")
    logger.info(f"  序列正则化权重: {args.sequence_reg_weight}")



    
    # 训练参数
    logger.info("【训练参数】")
    logger.info(f"  训练轮数: {args.epochs}")
    logger.info(f"  学习率: {args.lr}")
    logger.info(f"  权重衰减: {args.weight_decay}")
    logger.info(f"  梯度裁剪: {args.grad_clip}")
    logger.info(f"  随机种子: {args.seed}")
    logger.info(f"  严格确定性模式: {'启用' if args.strict_deterministic else '禁用'}")
    if args.strict_deterministic:
        logger.info("    - 单线程执行模式")
        logger.info("    - 严格CUDA确定性算法")
        logger.info("    - 可能影响训练性能")
    
    # 模态丢弃参数
    logger.info("【模态丢弃参数】")
    logger.info(f"  启用模态丢弃: {args.enable_modal_dropout}")
    logger.info(f"  模态丢弃概率阈值: {args.modal_dropout_prob} (随机数>此值时丢弃一种模态)")
    logger.info(f"  训练时应用模态丢弃: {args.enable_modal_dropout}")
    logger.info(f"  测试时应用模态丢弃: False")   # 测试时不使用模态丢弃
    
    # 其他参数
    logger.info("【其他参数】")
    logger.info(f"  保存目录: {args.save_dir}")
    logger.info(f"  日志目录: {args.log_dir}")
    logger.info(f"  恢复检查点: {args.resume if args.resume else 'None'}")
    logger.info(f"  仅测试模式: {args.test_only}")
    logger.info(f"  仅评估模式: {args.eval_only}")
    logger.info(f"  静默模式: {args.quiet}")
    logger.info(f"  CUDA调试: {args.debug_cuda}")
    
    logger.info("=" * 80)
    logger.info("使用数值稳定版本的训练代码 + 完整确定性修复")
    logger.info("已修复: torch.randn随机性 + DataLoader随机性 + 模型初始化随机性")
    
    # 设置TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载数据
    logger.info('Loading data...')
    try:
        train_loader, test_loader, road_network, normalization_stats = get_trifusion_loaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            route_min_len=args.route_min_len,
            route_max_len=args.route_max_len,
            gps_min_len=args.gps_min_len,
            gps_max_len=args.gps_max_len,
            cdr_min_len=args.cdr_min_len,
            cdr_max_len=args.cdr_max_len,
            num_samples=args.num_samples,
            test_ratio=args.test_ratio,
            modal_dropout_prob=args.modal_dropout_prob,
            apply_modal_dropout_train=False,  # 训练时默认不丢弃模态
            apply_modal_dropout_test=False,   # 测试时默认不丢弃模态
            seed=args.seed
        )
        logger.info("=" * 60)
        logger.info("数据加载详情")
        logger.info("=" * 60)
        logger.info(f'数据加载成功: {len(train_loader.dataset)} 训练样本, {len(test_loader.dataset)} 测试样本')
        logger.info(f'实际训练批次数: {len(train_loader)}')
        logger.info(f'实际测试批次数: {len(test_loader)}')
        logger.info(f'训练数据shuffle: True (使用确定性generator)')
        logger.info(f'测试数据shuffle: False')
        logger.info(f'DataLoader worker数量: {args.num_workers}')
        logger.info(f'DataLoader确定性控制: 已启用 (generator + worker_init_fn)')
        
        # 记录标准化统计信息
        logger.info('【标准化统计信息】')
        logger.info(f'  GPS: 经度 mean={normalization_stats["gps_mean_lon"]:.6f}, std={normalization_stats["gps_std_lon"]:.6f}')
        logger.info(f'       纬度 mean={normalization_stats["gps_mean_lat"]:.6f}, std={normalization_stats["gps_std_lat"]:.6f}')
        logger.info(f'       样本数量: {normalization_stats["gps_sample_count"]}')
        logger.info(f'  CDR: 经度 mean={normalization_stats["cdr_mean_lon"]:.6f}, std={normalization_stats["cdr_std_lon"]:.6f}')
        logger.info(f'       纬度 mean={normalization_stats["cdr_mean_lat"]:.6f}, std={normalization_stats["cdr_std_lat"]:.6f}')
        logger.info(f'       样本数量: {normalization_stats["cdr_sample_count"]}')
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    # 获取路网规模
    road_network_size = road_network.x.size(0)
    logger.info(f'路网规模: {road_network_size} 路段')
    
    # 如果需要从检查点恢复，先加载标准化统计信息
    checkpoint_normalization_stats = None
    if args.resume and os.path.isfile(args.resume):
        try:
            logger.info(f'预加载检查点以获取标准化统计信息: {args.resume}')
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'normalization_stats' in checkpoint:
                checkpoint_normalization_stats = checkpoint['normalization_stats']
                logger.info('从检查点恢复标准化统计信息')
                logger.info(f'  检查点GPS: 经度 mean={checkpoint_normalization_stats["gps_mean_lon"]:.6f}, std={checkpoint_normalization_stats["gps_std_lon"]:.6f}')
                logger.info(f'            纬度 mean={checkpoint_normalization_stats["gps_mean_lat"]:.6f}, std={checkpoint_normalization_stats["gps_std_lat"]:.6f}')
                logger.info(f'  检查点CDR: 经度 mean={checkpoint_normalization_stats["cdr_mean_lon"]:.6f}, std={checkpoint_normalization_stats["cdr_std_lon"]:.6f}')
                logger.info(f'            纬度 mean={checkpoint_normalization_stats["cdr_mean_lat"]:.6f}, std={checkpoint_normalization_stats["cdr_std_lat"]:.6f}')
            else:
                logger.warning('检查点中没有标准化统计信息，将使用当前数据集的统计信息')
        except Exception as e:
            logger.error(f"预加载检查点失败: {e}")
    
    # 选择使用哪个标准化统计信息
    final_normalization_stats = checkpoint_normalization_stats if checkpoint_normalization_stats is not None else normalization_stats
    
    # 计算静态输入-目标指标（只在训练开始时计算一次作为基准）
    try:
        logger.info('计算测试集静态输入-目标基准指标...')
        static_baseline_metrics = calculate_static_input_target_metrics(test_loader, device, logger, eos_token_id=-1)
        logger.info(f'测试集静态基准指标: {static_baseline_metrics}')
    except Exception as e:
        logger.error(f"静态基准指标计算失败: {e}")
        static_baseline_metrics = {}

    # 初始化模型
    logger.info('初始化模型...')
    try:
        # 在模型初始化前再次确保种子设置
        set_seed(args.seed)
        
        model = TrajFuse(
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            road_network_size=road_network_size,
            dropout=args.dropout,
            mlm_loss_weight=args.mlm_loss_weight,
            mlm_mask_prob=args.mlm_mask_prob,
            mlm_replace_prob=args.mlm_replace_prob,
            mlm_random_prob=args.mlm_random_prob,
            main_loss_weight=args.main_loss_weight,
            contrastive_loss_weight=args.contrastive_loss_weight,  
            contrastive_temperature=args.contrastive_temperature, 
            sequence_reg_weight=args.sequence_reg_weight,
            generation_mode='parallel',
            normalization_stats=final_normalization_stats,
            deterministic_seed=args.seed
        )
        model = model.to(device)
        
        # 详细记录模型配置
        logger.info("=" * 60)
        logger.info("模型初始化详情")
        logger.info("=" * 60)
        logger.info(f'模型类型: TrajFuse')
        logger.info(f'生成模式: parallel (并行生成)')
        logger.info(f'总参数量: {sum(p.numel() for p in model.parameters()):,}')
        logger.info(f'可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
        logger.info(f'路网规模: {road_network_size} 路段')
        logger.info(f'确定性种子: {args.seed}')
        logger.info(f'数值稳定性设置: eps=1e-8, clip_value=10.0')
        
        # 记录模型各组件参数量
        total_params = 0
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            total_params += module_params
            logger.info(f'  {name}: {module_params:,} 参数')
        
        logger.info(f'设备: {device}')
        logger.info(f'混合精度: 未启用')
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return
    
    # 定义损失函数和优化器
    logger.info("=" * 60)
    logger.info("优化器和调度器配置")
    logger.info("=" * 60)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    logger.info(f'损失函数: CrossEntropyLoss (ignore_index=-1)')
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        eps=1e-8,  
        amsgrad=True  
    )
    logger.info(f'优化器: AdamW')
    logger.info(f'  学习率: {args.lr}')
    logger.info(f'  权重衰减: {args.weight_decay}')
    logger.info(f'  eps: 1e-8 (数值稳定性)')
    logger.info(f'  amsgrad: True')
    
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7
    )
    logger.info(f'学习率调度器: ReduceLROnPlateau')
    logger.info(f'  模式: min (监控验证损失)')
    logger.info(f'  减少因子: 0.5')
    logger.info(f'  耐心值: 3 epochs')
    logger.info(f'  最小学习率: 1e-7')
    logger.info("=" * 60)
    
    # 从检查点恢复
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            try:
                logger.info(f'加载检查点: {args.resume}')
                checkpoint = torch.load(args.resume, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint['best_val_loss']
                logger.info(f'从epoch {start_epoch}恢复，最佳验证损失 {best_val_loss:.4f}')
            except Exception as e:
                logger.error(f"检查点加载失败: {e}")
                return
        else:
            logger.error(f'检查点文件不存在: {args.resume}')
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 只进行eval数据加载测试
    if args.eval_only:
        logger.info('=== EVAL数据诊断模式 ===')
        logger.info('跳过训练，仅测试eval数据加载...')
        logger.info('注意：将使用新初始化的模型参数，不加载已保存的模型')
        
        # 设置eval数据路径
        eval_data_path = "/root/autodl-tmp/data/chengdu/chengdu_eval_extension65_50_35_road2gps.pkl"
        logger.info(f'EVAL诊断模式：尝试加载eval数据集: {eval_data_path}')
        
        # 检查数据文件是否存在
        if not os.path.exists(eval_data_path):
            logger.error(f'EVAL数据文件不存在: {eval_data_path}')
            return
        
        logger.info(f'EVAL数据文件存在，大小: {os.path.getsize(eval_data_path) / 1024 / 1024:.2f} MB')
        
        try:
            # 步骤1：直接读取pkl文件查看基本信息
            logger.info('步骤1：直接读取pkl文件...')
            import pickle
            with open(eval_data_path, 'rb') as f:
                raw_data = pickle.load(f)
            
            if hasattr(raw_data, 'shape'):
                logger.info(f'原始eval数据形状: {raw_data.shape}')
                logger.info(f'数据列: {list(raw_data.columns) if hasattr(raw_data, "columns") else "N/A"}')
            else:
                logger.info(f'原始eval数据类型: {type(raw_data)}')
                if hasattr(raw_data, '__len__'):
                    logger.info(f'数据长度: {len(raw_data)}')
            
            # 检查数据范围
            if hasattr(raw_data, 'columns') and 'sparse_lat_list' in raw_data.columns and 'sparse_lng_list' in raw_data.columns:
                # 计算实际数据范围
                all_lats = []
                all_lngs = []
                for idx in range(min(100, len(raw_data))):  # 检查前100个样本
                    lat_list = raw_data.iloc[idx]['sparse_lat_list']
                    lng_list = raw_data.iloc[idx]['sparse_lng_list']
                    all_lats.extend(lat_list)
                    all_lngs.extend(lng_list)
                
                if all_lats and all_lngs:
                    min_lat, max_lat = min(all_lats), max(all_lats)
                    min_lng, max_lng = min(all_lngs), max(all_lngs)
                    logger.info(f'实际数据范围:')
                    logger.info(f'  纬度范围: {min_lat:.6f} ~ {max_lat:.6f}')
                    logger.info(f'  经度范围: {min_lng:.6f} ~ {max_lng:.6f}')
                    logger.info(f'期望数据范围:')
                    logger.info(f'  纬度范围: 30.652830 ~ 30.726490')
                    logger.info(f'  经度范围: 104.042110 ~ 104.129070')
                    
                    # 检查是否在期望范围内
                    lat_in_range = 30.652830 <= min_lat and max_lat <= 30.726490
                    lng_in_range = 104.042110 <= min_lng and max_lng <= 104.129070
                    logger.info(f'数据范围检查: 纬度{"✓" if lat_in_range else "✗"}, 经度{"✓" if lng_in_range else "✗"}')
            
            #尝试使用get_trifusion_loaders加载数据
            logger.info('步骤2：尝试使用get_trifusion_loaders加载eval数据...')
            _, eval_loader, eval_road_network, eval_normalization_stats = get_trifusion_loaders(
                data_path=eval_data_path,
                batch_size=args.batch_size,
                num_workers=0,  # 设为0避免多进程问题
                route_min_len=args.route_min_len,
                route_max_len=args.route_max_len,
                gps_min_len=args.gps_min_len,
                gps_max_len=args.gps_max_len,
                cdr_min_len=args.cdr_min_len,
                cdr_max_len=args.cdr_max_len,
                num_samples=None,  # 使用全部样本
                test_ratio=1.0,  # 全部作为测试数据
                modal_dropout_prob=0.0,  # 测试时不使用modal dropout
                apply_modal_dropout_train=False,
                apply_modal_dropout_test=False,
                seed=args.seed
            )
            
            logger.info(f'EVAL数据加载成功: {len(eval_loader.dataset)} 个样本')
            logger.info(f'EVAL路网规模: {eval_road_network.x.size(0)} 个节点')
            
            # 步骤3：尝试迭代数据加载器
            logger.info('步骤3：测试数据加载器迭代...')
            batch_count = 0
            error_count = 0
            total_samples = 0
            
            try:
                for batch_idx, batch in enumerate(eval_loader):
                    batch_count += 1
                    
                    # 检查batch的基本信息
                    if batch_idx < 3:  # 只详细检查前3个batch
                        logger.info(f'  Batch {batch_idx}:')
                        logger.info(f'    gps_data: {batch["gps_data"].shape if batch["gps_data"] is not None else "None"}')
                        logger.info(f'    cdr_data: {batch["cdr_data"].shape if batch["cdr_data"] is not None else "None"}')
                        logger.info(f'    road_gps_data: {batch["road_gps_data"].shape if batch["road_gps_data"] is not None else "None"}')
                        logger.info(f'    target_seg_data: {batch["target_seg_data"].shape}')
                        logger.info(f'    road_network: {batch["road_network"].x.shape}')
                        
                        # 检查数据有效性
                        if batch["target_seg_data"] is not None:
                            max_id = batch["target_seg_data"].max().item()
                            road_network_size = batch["road_network"].x.size(0)
                            logger.info(f'    最大路段ID: {max_id}, 路网大小: {road_network_size}')
                            if max_id >= road_network_size:
                                logger.warning(f'    ⚠️  batch {batch_idx} 路段ID超出范围!')
                                error_count += 1
                    
                    total_samples += batch["target_seg_data"].size(0)
                    
                    # 限制检查批次数以避免耗时过长
                    if batch_count >= 10:
                        logger.info(f'  已检查 {batch_count} 个batch，停止详细检查...')
                        break
                        
            except Exception as e:
                logger.error(f'  数据加载器迭代失败: {e}')
                error_count += 1
            
            logger.info(f'数据加载器测试完成: 检查了 {batch_count} 个batch, {total_samples} 个样本, {error_count} 个错误')
            
            # 步骤4：如果加载成功，尝试初始化模型并运行一个小测试
            if error_count == 0:
                logger.info('步骤4：初始化新模型并进行简单测试...')
                
                # 检查路网一致性
                if eval_road_network.x.size(0) != road_network.x.size(0):
                    logger.warning(f'EVAL数据集的路网规模({eval_road_network.x.size(0)})与训练数据集({road_network.x.size(0)})不一致')
                    # 对于eval诊断模式，我们使用eval数据的路网来初始化模型
                    eval_road_network_size = eval_road_network.x.size(0)
                    logger.info(f'EVAL诊断模式：使用eval数据的路网规模 {eval_road_network_size} 初始化模型')
                else:
                    eval_road_network_size = road_network.x.size(0)
                    logger.info(f'EVAL诊断模式：路网规模一致，使用规模 {eval_road_network_size} 初始化模型')
                
                # 初始化新的模型（使用eval数据的标准化统计信息和路网规模）
                try:
                    # 确保确定性初始化
                    set_seed(args.seed)
                    
                    eval_model = TrajFuse(
                        d_model=args.d_model,
                        nhead=args.nhead,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_decoder_layers,
                        dim_feedforward=args.dim_feedforward,
                        road_network_size=eval_road_network_size,
                        dropout=args.dropout,
                        mlm_loss_weight=args.mlm_loss_weight,
                        mlm_mask_prob=args.mlm_mask_prob,
                        mlm_replace_prob=args.mlm_replace_prob,
                        mlm_random_prob=args.mlm_random_prob,
                        main_loss_weight=args.main_loss_weight,
                        contrastive_loss_weight=args.contrastive_loss_weight,  
                        contrastive_temperature=args.contrastive_temperature,  
                        sequence_reg_weight=args.sequence_reg_weight,
                        generation_mode='parallel',
                        normalization_stats=eval_normalization_stats,
                        deterministic_seed=args.seed  # 传递确定性种子
                    )
                    eval_model = eval_model.to(device)
                    logger.info(f'EVAL诊断模式：新模型初始化完成，参数量: {sum(p.numel() for p in eval_model.parameters())}')
                    
                    # 尝试运行一个小batch的前向传播
                    eval_model.eval()
                    test_batch_count = 0
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(eval_loader):
                            if batch_idx >= 1:  # 只测试1个batch
                                break
                            
                            try:
                                logger.info(f'  测试batch {batch_idx} 的模型前向传播...')
                                
                                # 将数据移到设备上
                                gps_data = batch['gps_data'].to(device) if batch['gps_data'] is not None else None
                                cdr_data = batch['cdr_data'].to(device) if batch['cdr_data'] is not None else None
                                road_seg_data = batch['road_gps_data'].to(device) if batch['road_gps_data'] is not None else None
                                target_seg_data = batch['target_seg_data'].to(device)
                                road_network_batch = batch['road_network'].to(device)
                                
                                # 提取扩展数据字段
                                kwargs = {}
                                if 'sparse_gps2route_data' in batch and batch['sparse_gps2route_data'] is not None:
                                    kwargs['sparse_gps2route_data'] = batch['sparse_gps2route_data'].to(device)
                                if 'cdr_gps2route_data' in batch and batch['cdr_gps2route_data'] is not None:
                                    kwargs['cdr_gps2route_data'] = batch['cdr_gps2route_data'].to(device)
                                if 'monitor_opath_data' in batch and batch['monitor_opath_data'] is not None:
                                    kwargs['monitor_opath_data'] = batch['monitor_opath_data'].to(device)
                                if 'merged_route_data' in batch and batch['merged_route_data'] is not None:
                                    kwargs['merged_route_data'] = batch['merged_route_data'].to(device)
                                if 'original_cpath_data' in batch and batch['original_cpath_data'] is not None:
                                    kwargs['original_cpath_data'] = batch['original_cpath_data'].to(device)
                                else:
                                    kwargs['original_cpath_data'] = target_seg_data
                                
                                # 提取扩展时间字段
                                if 'gps_time_data' in batch and batch['gps_time_data'] is not None:
                                    kwargs['gps_time_data'] = batch['gps_time_data'].to(device)
                                if 'cdr_time_data' in batch and batch['cdr_time_data'] is not None:
                                    kwargs['cdr_time_data'] = batch['cdr_time_data'].to(device)
                                if 'road_gps_time_data' in batch and batch['road_gps_time_data'] is not None:
                                    kwargs['road_time_data'] = batch['road_gps_time_data'].to(device)

                                # 模型前向传播
                                pred_segments, loss_dict, _ = eval_model(
                                    gps_data=gps_data,
                                    cdr_data=cdr_data,
                                    road_seg_data=road_seg_data,
                                    road_network=road_network_batch,
                                    target=target_seg_data,
                                    **kwargs
                                )
                                
                                logger.info(f'  ✓ 模型前向传播成功')
                                logger.info(f'    pred_segments形状: {pred_segments.shape}')
                                logger.info(f'    损失: {loss_dict["total_loss"].item():.4f}')
                                test_batch_count += 1
                                
                            except Exception as e:
                                logger.error(f'  ✗ 模型前向传播失败: {e}')
                                import traceback
                                logger.error(f'  详细错误信息:\n{traceback.format_exc()}')
                    
                    if test_batch_count > 0:
                        logger.info('EVAL数据诊断成功：数据加载和模型推理都正常!')
                    else:
                        logger.error('❌ EVAL数据诊断部分成功：数据加载正常但模型推理失败')
                        
                except Exception as e:
                    logger.error(f'EVAL诊断模式：模型初始化失败: {e}')
                    import traceback
                    logger.error(f'详细错误信息:\n{traceback.format_exc()}')
            else:
                logger.error('❌ EVAL数据诊断失败：数据加载存在问题')
                
        except Exception as e:
            logger.error(f'EVAL数据加载过程中发生错误: {e}')
            import traceback
            logger.error(f'详细错误信息:\n{traceback.format_exc()}')
        
        return
    
    # 只进行测试
    if args.test_only:
        if args.resume:
            logger.info('仅运行测试...')
            
            # 加载独立的测试数据集
            test_only_data_path = "/root/autodl-tmp/data/chengdu/chengdu_eval_extension65_50_35_road2gps.pkl"
            logger.info(f'测试模式：加载独立测试数据集: {test_only_data_path}')
            
            try:
                # 创建独立的测试数据加载器
                _, test_only_loader, test_only_road_network, test_only_normalization_stats = get_trifusion_loaders(
                    data_path=test_only_data_path,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    route_min_len=args.route_min_len,
                    route_max_len=args.route_max_len,
                    gps_min_len=args.gps_min_len,
                    gps_max_len=args.gps_max_len,
                    cdr_min_len=args.cdr_min_len,
                    cdr_max_len=args.cdr_max_len,
                    num_samples=None,  # 使用全部样本
                    test_ratio=1.0,  # 全部作为测试数据
                    modal_dropout_prob=0.0,  # 测试时不使用modal dropout
                    apply_modal_dropout_train=False,
                    apply_modal_dropout_test=False,
                    seed=args.seed
                )
                logger.info(f'测试模式：独立测试数据加载成功: {len(test_only_loader.dataset)} 测试样本')
                
                # 检查路网一致性
                if test_only_road_network.x.size(0) != road_network.x.size(0):
                    logger.warning(f'测试模式：独立测试数据集的路网规模({test_only_road_network.x.size(0)})与训练数据集({road_network.x.size(0)})不一致')
                
                # 使用独立测试数据集进行测试
                test_results = test(model, test_only_loader, device, logger, args.save_dir)
                logger.info(f'独立测试集结果: {test_results}')
                
                # 计算测试集的模型生成指标
                try:
                    logger.info('测试模式：计算独立测试集模型生成指标...')
                    test_generation_metrics = calculate_model_generation_metrics(model, test_only_loader, device, logger, eos_token_id=-1)
                    logger.info(f'独立测试集生成指标: rid_acc={test_generation_metrics.get("rid_acc", 0.0):.4f}, '
                               f'rid_recall={test_generation_metrics.get("rid_recall", 0.0):.4f}, '
                               f'rid_precision={test_generation_metrics.get("rid_precision", 0.0):.4f}, '
                               f'rid_f1={test_generation_metrics.get("rid_f1", 0.0):.4f}')
                    
                    # 将生成指标添加到测试结果中
                    for key, value in test_generation_metrics.items():
                        test_results[f'gen_{key}'] = value
                    logger.info(f'完整独立测试结果 (包含生成指标): {test_results}')
                except Exception as e:
                    logger.error(f"独立测试集生成指标计算失败: {e}")
                    
                # 计算独立测试集的静态输入-目标基准指标
                try:
                    logger.info('测试模式：计算独立测试集静态输入-目标基准指标...')
                    test_static_baseline_metrics = calculate_static_input_target_metrics(test_only_loader, device, logger, eos_token_id=-1)
                    logger.info(f'独立测试集静态基准指标: {test_static_baseline_metrics}')
                    
                    # 将静态基准指标添加到测试结果中
                    for key, value in test_static_baseline_metrics.items():
                        test_results[f'static_{key}'] = value
                except Exception as e:
                    logger.error(f"独立测试集静态基准指标计算失败: {e}")
                    
            except Exception as e:
                logger.error(f"测试模式：独立测试数据加载失败: {e}")
                logger.info('测试模式：回退到使用原始测试数据集...')
                
                # 回退到原始测试数据集
                test_results = test(model, test_loader, device, logger, args.save_dir)
                logger.info(f'原始测试集结果: {test_results}')
                
                # 计算原始测试集的模型生成指标
                try:
                    logger.info('测试模式：计算原始测试集模型生成指标...')
                    test_generation_metrics = calculate_model_generation_metrics(model, test_loader, device, logger, eos_token_id=-1)
                    logger.info(f'原始测试集生成指标: rid_acc={test_generation_metrics.get("rid_acc", 0.0):.4f}, '
                               f'rid_recall={test_generation_metrics.get("rid_recall", 0.0):.4f}, '
                               f'rid_precision={test_generation_metrics.get("rid_precision", 0.0):.4f}, '
                               f'rid_f1={test_generation_metrics.get("rid_f1", 0.0):.4f}')
                    
                    # 将生成指标添加到测试结果中
                    for key, value in test_generation_metrics.items():
                        test_results[f'gen_{key}'] = value
                    logger.info(f'完整原始测试结果 (包含生成指标): {test_results}')
                except Exception as e:
                    logger.error(f"原始测试集生成指标计算失败: {e}")
            
            return
        else:
            logger.error('测试模式必须提供检查点')
            return
    
    # 记录系统环境信息
    logger.info("=" * 60)
    logger.info("训练环境信息")
    logger.info("=" * 60)
    logger.info(f'PyTorch版本: {torch.__version__}')
    logger.info(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logger.info(f'CUDA版本: {torch.version.cuda}')
        logger.info(f'GPU数量: {torch.cuda.device_count()}')
        logger.info(f'当前GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    logger.info(f'cuDNN确定性: {torch.backends.cudnn.deterministic}')
    logger.info(f'cuDNN基准测试: {torch.backends.cudnn.benchmark}')
    
    import platform
    logger.info(f'操作系统: {platform.system()} {platform.release()}')
    logger.info(f'Python版本: {platform.python_version()}')
    
    # 记录确定性设置总结
    logger.info("【确定性设置总结】")
    logger.info("全局随机种子: 已设置")
    logger.info("DataLoader确定性: 已启用 (generator + worker_init_fn)")
    logger.info("模型内部随机性: 已消除 (torch.randn -> torch.zeros)")
    logger.info("模型初始化: 确定性种子")
    logger.info("cuDNN确定性: 已启用")
    if args.strict_deterministic:
        logger.info("严格确定性模式: 已启用 (单线程 + 严格CUDA)")
        logger.info("训练结果完全可重复性: 已保证 (可能影响性能)")
    else:
        logger.info("标准确定性模式: 已启用 (允许并行计算)")
        logger.info("训练结果可重复性: 已保证")
    logger.info("=" * 60)
    
    # 训练循环
    logger.info('开始训练...')
    for epoch in range(start_epoch, args.epochs):
        try:
            # 训练一个epoch
            start_time = time.time()
            
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, writer, args.grad_clip, logger,
                enable_modal_dropout=args.enable_modal_dropout, modal_dropout_prob=args.modal_dropout_prob
            )
            epoch_time = time.time() - start_time
            
            # 检查训练是否成功
            if train_metrics['success_rate'] < 0.5:
                logger.warning(f"Epoch {epoch} 训练成功率过低: {train_metrics['success_rate']:.2%}")
            
            # 验证
            val_metrics_dict = validate(model, test_loader, criterion, device, logger)
            
            # 检查验证是否成功
            if val_metrics_dict['success_rate'] < 0.5:
                logger.warning(f"Epoch {epoch} 验证成功率过低: {val_metrics_dict['success_rate']:.2%}")
            
            # 计算训练集模型生成指标
            train_generation_metrics = {}
            try:
                logger.info(f'Epoch {epoch}: 计算训练集模型生成指标...')
                train_generation_metrics = calculate_model_generation_metrics(model, train_loader, device, logger, eos_token_id=-1)
                logger.info(f'Epoch {epoch} 训练集生成指标: rid_acc={train_generation_metrics.get("rid_acc", 0.0):.4f}, '
                           f'rid_recall={train_generation_metrics.get("rid_recall", 0.0):.4f}, '
                           f'rid_precision={train_generation_metrics.get("rid_precision", 0.0):.4f}, '
                           f'rid_f1={train_generation_metrics.get("rid_f1", 0.0):.4f}')
            except Exception as e:
                logger.error(f"Epoch {epoch} 训练集生成指标计算失败: {e}")
                train_generation_metrics = {}
            
            # 计算测试集模型生成指标
            test_generation_metrics = {}
            try:
                logger.info(f'Epoch {epoch}: 计算测试集模型生成指标...')
                test_generation_metrics = calculate_model_generation_metrics(model, test_loader, device, logger, eos_token_id=-1)
                logger.info(f'Epoch {epoch} 测试集生成指标: rid_acc={test_generation_metrics.get("rid_acc", 0.0):.4f}, '
                           f'rid_recall={test_generation_metrics.get("rid_recall", 0.0):.4f}, '
                           f'rid_precision={test_generation_metrics.get("rid_precision", 0.0):.4f}, '
                           f'rid_f1={test_generation_metrics.get("rid_f1", 0.0):.4f}')
            except Exception as e:
                logger.error(f"Epoch {epoch} 测试集生成指标计算失败: {e}")
                test_generation_metrics = {}
            
            # 更新学习率
            if not math.isinf(val_metrics_dict['loss']) and not math.isnan(val_metrics_dict['loss']):
                scheduler.step(val_metrics_dict['loss'])
            else:
                logger.warning(f"Epoch {epoch} 验证损失无效，跳过学习率调整")
            
            # 记录到TensorBoard
            if writer:
                writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
                writer.add_scalar('epoch/val_loss', val_metrics_dict['loss'], epoch)
                writer.add_scalar('epoch/train_main_loss', train_metrics['main_loss'], epoch)
                writer.add_scalar('epoch/val_main_loss', val_metrics_dict['main_loss'], epoch)
                writer.add_scalar('epoch/train_contrastive_temperature', train_metrics['contrastive_temperature'], epoch)
                writer.add_scalar('epoch/val_contrastive_temperature', val_metrics_dict['contrastive_temperature'], epoch)
                writer.add_scalar('epoch/train_mlm_loss', train_metrics['mlm_loss'], epoch)
                writer.add_scalar('epoch/val_mlm_loss', val_metrics_dict['mlm_loss'], epoch)
                writer.add_scalar('epoch/train_contrastive_loss', train_metrics['contrastive_loss'], epoch)
                writer.add_scalar('epoch/val_contrastive_loss', val_metrics_dict['contrastive_loss'], epoch)
                writer.add_scalar('epoch/train_sequence_reg', train_metrics['sequence_reg'], epoch)
                writer.add_scalar('epoch/val_sequence_reg', val_metrics_dict['sequence_reg'], epoch)
                writer.add_scalar('epoch/train_success_rate', train_metrics['success_rate'], epoch)
                writer.add_scalar('epoch/val_success_rate', val_metrics_dict['success_rate'], epoch)
                writer.add_scalar('epoch/val_rid_acc', val_metrics_dict.get('rid_acc', 0.0), epoch)
                writer.add_scalar('epoch/val_lcs_f1', val_metrics_dict.get('lcs_f1', 0.0), epoch)
                writer.add_scalar('epoch/val_trad_acc', val_metrics_dict.get('trad_acc', 0.0), epoch)
                
                
                # 记录训练集生成指标
                for key, value in train_generation_metrics.items():
                    if key in ['rid_acc', 'rid_recall', 'rid_precision', 'rid_f1', 'success_rate']:
                        writer.add_scalar(f'epoch/train_gen_{key}', value, epoch)
                
                # 记录测试集生成指标
                for key, value in test_generation_metrics.items():
                    if key in ['rid_acc', 'rid_recall', 'rid_precision', 'rid_f1', 'success_rate']:
                        writer.add_scalar(f'epoch/test_gen_{key}', value, epoch)
            
            # 记录到日志
            log_entry = (
                f"Epoch {epoch} - 时间: {epoch_time:.1f}s - "
                f"训练损失: {train_metrics['loss']:.4f} (主: {train_metrics['main_loss']:.4f}, 温度: {train_metrics['contrastive_temperature']:.4f}, MLM: {train_metrics['mlm_loss']:.4f}, 对比: {train_metrics['contrastive_loss']:.4f}, 正则: {train_metrics['sequence_reg']:.4f}) "
                f"(成功率: {train_metrics['success_rate']:.2%}), "
                f"验证损失: {val_metrics_dict['loss']:.4f} (主: {val_metrics_dict['main_loss']:.4f}, 温度: {val_metrics_dict['contrastive_temperature']:.4f}, MLM: {val_metrics_dict['mlm_loss']:.4f}, 对比: {val_metrics_dict['contrastive_loss']:.4f}, 正则: {val_metrics_dict['sequence_reg']:.4f}) "
                f"(成功率: {val_metrics_dict['success_rate']:.2%}), "
                f"Val RidAcc: {val_metrics_dict.get('rid_acc', 0.0):.4f}, "
                f"Val TradAcc: {val_metrics_dict.get('trad_acc', 0.0):.4f}, "
                f"TrainGen RidAcc: {train_generation_metrics.get('rid_acc', 0.0):.4f}, "
                f"TestGen RidAcc: {test_generation_metrics.get('rid_acc', 0.0):.4f}, "
                f"学习率: {optimizer.param_groups[0]['lr']:.6f}"
            )
            logger.info(log_entry)
            
            # 保存最佳模型（基于验证损失和成功率）
            is_best = (
                val_metrics_dict['loss'] < best_val_loss and 
                val_metrics_dict['success_rate'] > 0.5 and
                not math.isinf(val_metrics_dict['loss']) and 
                not math.isnan(val_metrics_dict['loss'])
            )
            
            if is_best:
                best_val_loss = val_metrics_dict['loss']
                
                try:
                    checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'normalization_stats': final_normalization_stats,  # 保存标准化统计信息
                    }
                    save_dict.update({k: v for k, v in val_metrics_dict.items() if k != 'loss'})
                    save_dict.update(static_baseline_metrics)  # 保存基准静态指标
                    # 添加训练集和测试集的生成指标
                    for key, value in train_generation_metrics.items():
                        save_dict[f'train_gen_{key}'] = value
                    for key, value in test_generation_metrics.items():
                        save_dict[f'test_gen_{key}'] = value
                    torch.save(save_dict, checkpoint_path)
                    
                    logger.info(f'保存最佳模型: 验证损失 {best_val_loss:.4f}, 成功率 {val_metrics_dict["success_rate"]:.2%}, '
                               f'训练生成RidAcc {train_generation_metrics.get("rid_acc", 0.0):.4f}, '
                               f'测试生成RidAcc {test_generation_metrics.get("rid_acc", 0.0):.4f}')
                except Exception as e:
                    logger.error(f"保存最佳模型失败: {e}")
            
            # 每10个epoch保存一次
            if epoch % 10 == 9:
                try:
                    checkpoint_path = os.path.join(args.save_dir, f'model_epoch_{epoch}.pt')
                    save_dict_epoch = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'current_val_loss': val_metrics_dict['loss'],
                        'normalization_stats': final_normalization_stats,  # 保存标准化统计信息
                    }
                    save_dict_epoch.update({k: v for k, v in val_metrics_dict.items()})
                    save_dict_epoch.update(static_baseline_metrics)  # 保存基准静态指标
                    # 添加训练集和测试集的生成指标
                    for key, value in train_generation_metrics.items():
                        save_dict_epoch[f'train_gen_{key}'] = value
                    for key, value in test_generation_metrics.items():
                        save_dict_epoch[f'test_gen_{key}'] = value
                    torch.save(save_dict_epoch, checkpoint_path)
                    logger.info(f'保存epoch {epoch}模型')
                except Exception as e:
                    logger.error(f"保存epoch模型失败: {e}")
            
        except Exception as e:
            logger.error(f"Epoch {epoch} 训练过程发生错误: {e}")
            continue
    
    # 训练完成，运行测试
    logger.info('训练完成，运行最终测试...')
    
    # 加载最佳模型
    best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'加载最佳模型: epoch {checkpoint["epoch"]}, 验证损失 {checkpoint["best_val_loss"]:.4f}')
        except Exception as e:
            logger.error(f"加载最佳模型失败: {e}")
    
    # 加载独立的测试数据集
    final_test_data_path = "/root/autodl-tmp/data/chengdu/chengdu_eval_extension65_50_35_road2gps.pkl"
    logger.info(f'加载独立测试数据集: {final_test_data_path}')
    
    try:
        # 创建独立的测试数据加载器
        _, final_test_loader, final_road_network, final_normalization_stats = get_trifusion_loaders(
            data_path=final_test_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            route_min_len=args.route_min_len,
            route_max_len=args.route_max_len,
            gps_min_len=args.gps_min_len,
            gps_max_len=args.gps_max_len,
            cdr_min_len=args.cdr_min_len,
            cdr_max_len=args.cdr_max_len,
            num_samples=None,  # 使用全部样本
            test_ratio=1.0,  # 全部作为测试数据
            modal_dropout_prob=0.0,  # 测试时不使用modal dropout
            apply_modal_dropout_train=False,
            apply_modal_dropout_test=False,
            seed=args.seed
        )
        logger.info(f'独立测试数据加载成功: {len(final_test_loader.dataset)} 测试样本')
        
        # 检查路网一致性
        if final_road_network.x.size(0) != road_network.x.size(0):
            logger.warning(f'独立测试数据集的路网规模({final_road_network.x.size(0)})与训练数据集({road_network.x.size(0)})不一致')
        
        # 使用独立测试数据集进行测试
        test_results_dict = test(model, final_test_loader, device, logger, args.save_dir, eos_token_id=-1)
        logger.info(f'独立测试集结果: {test_results_dict}')
        
        # 计算最终测试集的模型生成指标
        try:
            logger.info('独立测试集：计算模型生成指标...')
            final_test_generation_metrics = calculate_model_generation_metrics(model, final_test_loader, device, logger, eos_token_id=-1)
            logger.info(f'独立测试集生成指标: rid_acc={final_test_generation_metrics.get("rid_acc", 0.0):.4f}, '
                       f'rid_recall={final_test_generation_metrics.get("rid_recall", 0.0):.4f}, '
                       f'rid_precision={final_test_generation_metrics.get("rid_precision", 0.0):.4f}, '
                       f'rid_f1={final_test_generation_metrics.get("rid_f1", 0.0):.4f}')
            
            # 将生成指标添加到测试结果中
            for key, value in final_test_generation_metrics.items():
                test_results_dict[f'final_gen_{key}'] = value
        except Exception as e:
            logger.error(f"独立测试集生成指标计算失败: {e}")
            
        # 计算独立测试集的静态输入-目标基准指标
        try:
            logger.info('计算独立测试集静态输入-目标基准指标...')
            final_static_baseline_metrics = calculate_static_input_target_metrics(final_test_loader, device, logger, eos_token_id=-1)
            logger.info(f'独立测试集静态基准指标: {final_static_baseline_metrics}')
            
            # 将静态基准指标添加到测试结果中
            for key, value in final_static_baseline_metrics.items():
                test_results_dict[f'final_{key}'] = value
        except Exception as e:
            logger.error(f"独立测试集静态基准指标计算失败: {e}")
            
    except Exception as e:
        logger.error(f"独立测试数据加载失败: {e}")
        logger.info('回退到使用原始测试数据集...')
        
        # 运行原始测试数据集的测试
        try:
            test_results_dict = test(model, test_loader, device, logger, args.save_dir, eos_token_id=-1)
            logger.info(f'原始测试集结果: {test_results_dict}')
            
            # 计算原始测试集的模型生成指标
            try:
                logger.info('原始测试集：计算模型生成指标...')
                final_test_generation_metrics = calculate_model_generation_metrics(model, test_loader, device, logger, eos_token_id=-1)
                logger.info(f'原始测试集生成指标: rid_acc={final_test_generation_metrics.get("rid_acc", 0.0):.4f}, '
                           f'rid_recall={final_test_generation_metrics.get("rid_recall", 0.0):.4f}, '
                           f'rid_precision={final_test_generation_metrics.get("rid_precision", 0.0):.4f}, '
                           f'rid_f1={final_test_generation_metrics.get("rid_f1", 0.0):.4f}')
                
                # 将生成指标添加到测试结果中
                for key, value in final_test_generation_metrics.items():
                    test_results_dict[f'final_gen_{key}'] = value
            except Exception as e:
                logger.error(f"原始测试集生成指标计算失败: {e}")
                
        except Exception as e:
            logger.error(f"原始测试过程失败: {e}")
    
    # 关闭TensorBoard写入器
    if writer:
        writer.close()

if __name__ == '__main__':
    import math
    main()