import sys
import torch
import numpy as np

sys.setrecursionlimit(500000)  # 设置递归深度限制以支持LCS计算

def memoize(fn):
    """返回输入函数的记忆化版本，缓存先前调用的结果"""
    cache = dict()
    def wrapped(*args, **kwargs):
        # 将list参数转换为tuple以便作为缓存键
        key_args = []
        for arg in args:
            if isinstance(arg, list):
                key_args.append(tuple(arg))
            else:
                key_args.append(arg)
        
        key = tuple(key_args)
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]
    return wrapped

@memoize
def lcs(xs, ys):
    """
    计算两个序列的最长公共子序列(LCS)
    
    参数:
        xs, ys: 输入序列
    返回:
        list: 最长公共子序列
    """
    @memoize
    def lcs_(i, j):
        if i and j:
            xe, ye = xs[i-1], ys[j-1]
            if xe == ye:
                return lcs_(i-1, j-1) + [xe]
            else:
                return max(lcs_(i, j-1), lcs_(i-1, j), key=len)
        else:
            return []
    return lcs_(len(xs), len(ys))

def shrink_seq(seq):
    """
    去除序列中连续重复的元素
    
    参数:
        seq: 输入序列
    返回:
        list: 去重后的序列
        
    示例:
        shrink_seq([1, 1, 2, 2, 2, 3]) -> [1, 2, 3]
    """
    if not seq:
        return []
    
    result = [seq[0]]
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            result.append(seq[i])
    return result

def cal_pre_recall_simple(batch_pre, batch_label, eos_token_id=0):
    """
    计算轨迹预测的3个核心指标（简化版）
    
    参数:
        batch_pre: list of lists, 预测的路段ID序列批次
        batch_label: list of lists, 真实的路段ID序列批次  
        eos_token_id: int, 序列结束标记ID（默认为1）
        
    返回:
        tuple: (rid_acc, rid_recall, rid_precision)
        - rid_acc: 路段ID准确率 - 逐位置精确匹配的比例
        - rid_recall: LCS召回率 - LCS长度/真实去重序列长度 (覆盖真实路径的程度)
        - rid_precision: LCS精确率 - LCS长度/预测去重序列长度 (预测路径的准确程度)
    """
    
    if not batch_pre or not batch_label or len(batch_pre) != len(batch_label):
        return 0.0, 0.0, 0.0
    
    batch_size = len(batch_pre)
    
    # 累积统计变量
    cnt = 0                    # 位置匹配的总数
    ttl = 0                    # 总的比较位置数
    ttl_trg_id_num = 0         # 去重真实序列的总长度
    ttl_pre_id_num = 0         # 去重预测序列的总长度  
    correct_id_num = 0         # LCS的总长度
    
    for num in range(batch_size):
        hmmm_list = list(batch_pre[num])      # 预测序列
        truth_list = list(batch_label[num])   # 真实序列
        
        # 1. 计算位置级准确率 (rid_acc)
        # 找到真实序列中第一个EOS的位置进行截断
        truth_eos_idx = len(truth_list)
        try:
            truth_eos_idx = truth_list.index(eos_token_id)
        except ValueError:
            pass  # 没有找到EOS，使用完整长度
            
        # 截断序列到EOS位置
        truncated_truth = truth_list[:truth_eos_idx] 
        truncated_pred = hmmm_list[:len(truncated_truth)]  # 预测序列截断到相同长度
        
        # 逐位置比较
        for i in range(min(len(truncated_pred), len(truncated_truth))):
            ttl += 1
            if truncated_pred[i] == truncated_truth[i]:
                cnt += 1
        
        # 2. 计算LCS相关指标 (rid_recall, rid_precision)
        # 对截断后的序列去重
        shr_trg_ids = shrink_seq(truncated_truth)
        shr_pre_ids = shrink_seq(truncated_pred)
        
        # 计算LCS长度
        lcs_length = len(lcs(shr_trg_ids, shr_pre_ids))
        
        # 累积统计
        correct_id_num += lcs_length
        ttl_trg_id_num += len(shr_trg_ids)
        ttl_pre_id_num += len(shr_pre_ids)
    
    # 计算最终指标
    rid_acc = cnt / ttl if ttl > 0 else 0.0
    rid_recall = correct_id_num / ttl_trg_id_num if ttl_trg_id_num > 0 else 0.0
    rid_precision = correct_id_num / ttl_pre_id_num if ttl_pre_id_num > 0 else 0.0
    
    return rid_acc, rid_recall, rid_precision

def cal_pre_recall_with_f1(batch_pre, batch_label, eos_token_id=1):
    """
    计算轨迹预测的4个核心指标（包含F1分数）
    
    返回:
        dict: 包含4个指标的字典
        - rid_acc: 路段ID准确率
        - rid_recall: LCS召回率  
        - rid_precision: LCS精确率
        - rid_f1: LCS F1分数
    """
    rid_acc, rid_recall, rid_precision = cal_pre_recall_simple(
        batch_pre, batch_label, eos_token_id
    )
    
    # 计算F1分数
    rid_f1 = (2 * rid_precision * rid_recall) / (rid_precision + rid_recall) \
             if (rid_precision + rid_recall) > 0 else 0.0
    
    return {
        'rid_acc': rid_acc,           # 路段ID准确率：逐位置精确匹配比例
        'rid_recall': rid_recall,     # LCS召回率：覆盖真实路径的程度
        'rid_precision': rid_precision, # LCS精确率：预测路径的准确程度  
        'rid_f1': rid_f1              # LCS F1分数：精确率和召回率的调和平均
    }

def print_metrics_simple(rid_acc, rid_recall, rid_precision):
    """
    格式化打印简化指标
    """
    print("=== 轨迹预测核心指标 ===")
    print(f"路段ID准确率 (rid_acc):     {rid_acc:.4f}")
    print(f"LCS召回率 (rid_recall):     {rid_recall:.4f}") 
    print(f"LCS精确率 (rid_precision):  {rid_precision:.4f}")
    print(f"LCS F1分数:                {2*rid_precision*rid_recall/(rid_precision+rid_recall) if (rid_precision+rid_recall)>0 else 0:.4f}")
    print("=" * 30)

# sample
if __name__ == '__main__':
    # 测试数据
    predictions = [
        [2, 3, 4, 5, 1, 6],    # 预测序列1，EOS=1
        [7, 8, 1, 9],          # 预测序列2，EOS=1
        [10, 11, 12]           # 预测序列3，无EOS
    ]
    
    targets = [
        [2, 3, 4, 4, 1, 7],    # 真实序列1，EOS=1
        [7, 8, 1],             # 真实序列2，EOS=1
        [10, 11, 13, 1]        # 真实序列3，EOS=1
    ]
    
    # 计算简化指标
    rid_acc, rid_recall, rid_precision = cal_pre_recall_simple(predictions, targets)
    print_metrics_simple(rid_acc, rid_recall, rid_precision)
    
    # 或者使用字典版本
    metrics = cal_pre_recall_with_f1(predictions, targets)
    print("\n=== 字典格式输出 ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}") 