import random
import torch
from matplotlib import pyplot as plt
import seaborn as sns
import kipoiseq
import numpy as np
import torch.distributed as dist
import time
from collections import defaultdict,deque
import datetime
from torchmetrics.functional import spearman_corrcoef,pearson_corrcoef

import torch.nn.functional as F


def norm_Adj(W):
    """
    Compute normalized adjacency matrix (row-normalized) for PyTorch tensors.
    
    Parameters
    ----------
    W : torch.Tensor
        Adjacency matrix, shape (N, N), can be on any device (CPU/GPU).
    
    Returns
    -------
    torch.Tensor
        Normalized adjacency matrix: (D_hat)^{-1} A_hat, same device as W.
    """
    assert W.shape[0] == W.shape[1], "Adjacency matrix must be square"
    
    N = W.shape[0]
    device = W.device
    
    # Add self-loops
    W = W + torch.eye(N, device=device)
    
    # Degree vector (row sums)
    deg = torch.sum(W, dim=1)  # shape (N,)
    
    # Row normalization: divide each row by its degree
    # Use broadcasting to avoid creating a full diagonal matrix
    norm_Adj = W / deg.unsqueeze(1)
    
    return norm_Adj


def asymmetric_loss_with_metrics(logits, targets, gamma_neg=0.5, gamma_pos=1.0, clip=0.05, reduction='mean'):
    """
    非对称损失函数 - 带召回率统计版本
    
    Returns:
        loss: 损失值
        metrics: 包含召回率等指标的字典
    """
    # 确保维度一致
    if logits.dim() == 1:
        logits = logits.unsqueeze(1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    
    # 计算概率和预测
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).float()
    
    # 计算召回率相关统计
    with torch.no_grad():
        # 正样本相关统计
        true_positives = ((predictions == 1) & (targets == 1)).sum().float()
        false_negatives = ((predictions == 0) & (targets == 1)).sum().float()
        total_positives = (targets == 1).sum().float()
        
        # 负样本相关统计（用于参考）
        true_negatives = ((predictions == 0) & (targets == 0)).sum().float()
        false_positives = ((predictions == 1) & (targets == 0)).sum().float()
        total_negatives = (targets == 0).sum().float()
        
        # 计算召回率（正样本）
        recall = (true_positives / (total_positives + 1e-8)).item()
        
        # 计算精确率（可选）
        precision = (true_positives / (predictions.sum() + 1e-8)).item()
        
        # 计算F1分数（可选）
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # 特别关注：在预测为正的样本中，有多少原本是负样本（可能被错误标记）
        false_positive_rate = (false_positives / (total_negatives + 1e-8)).item()
        
        metrics = {
            'recall': recall,                    # 正样本召回率
            'precision': precision,              # 精确率
            'f1': f1,                            # F1分数
            'true_positives': true_positives.item(),
            'false_negatives': false_negatives.item(),
            'false_positives': false_positives.item(),
            'false_positive_rate': false_positive_rate,  # 假阳性率
            'total_positives': total_positives.item(),
            'total_negatives': total_negatives.item()
        }
    
    # 裁剪概率防止数值不稳定
    probs = torch.clamp(probs, clip, 1 - clip)
    
    # 计算交叉熵
    pos_loss = -targets * torch.log(probs)
    neg_loss = -(1 - targets) * torch.log(1 - probs)
    
    # 应用非对称权重
    pos_loss = gamma_pos * pos_loss
    neg_loss = gamma_neg * neg_loss
    
    # 组合损失
    loss = pos_loss + neg_loss
    
    # 返回指定格式的损失
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:  # 'none'
        loss = loss.squeeze()
    print(metrics)
    return loss

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)



class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def initialize_parameters(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight) # Xavier 均匀分布初始化


def plot_tracks_comparision(tracks, index,pearsonr,spearman,p_value,interval=None, height=1.5):
    # tracks : {"track1":np.array([...])} 
    # interval : "chr1:1-10000"


    if interval == None:
        plot_interval = False
        n = [i for i in tracks.values()][0]
        interval= kipoiseq.Interval('xx', 0,len(n))
    else:
        plot_interval = True
        start=interval.split(":")[1].split("-")[0]
        end=interval.split(":")[1].split("-")[1]
        chr_=interval.split(":")[0]
        interval = kipoiseq.Interval(chr_, start,end)

    fig, axes = plt.subplots(1, 1, figsize=(20, height * len(tracks)), sharex=True)
    ax = axes
    for (title, y) in tracks.items():
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y,alpha=0.5, label = title)
        # ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
        if plot_interval == True: 
            ax.set_xlabel(str(interval))
        plt.tight_layout()
        plt.title(''+list(tracks.keys())[0]+f' pearsonr corr:{pearsonr:.3f} , spearman corr:{spearman:.3f}')
    plt.legend()
    plt.savefig(f'plot/bbs{index}.jpg')









def compute_correlations_total(preds, targets):

    
    result= {
        'Pearson_Correlation': pearson_corrcoef(preds.flatten(), targets.flatten()).item(),
        'Spearman_Correlation': spearman_corrcoef(preds.flatten(), targets.flatten()).item()
    }

    return result

def compute_correlations_individual(preds, targets):
    spearman_list,pearson_list = [],[]

    for i in range(preds.shape[0]):

            pearson_corr=pearson_corrcoef(preds[i], targets[i])
            spearman_corrc=spearman_corrcoef(preds[i], targets[i])
            spearman_list.append(spearman_corrc)
            pearson_list.append(pearson_corr)
            

    return {
        'Pearson_Correlation':sum(pearson_list) / preds.shape[0],
        'Spearman_Correlation':sum(spearman_list) / preds.shape[0]
    }



def getIOU(logits:torch.Tensor,groundTruth:torch.Tensor):

    max_sum=torch.max(logits,groundTruth).sum(axis=-1)
    neg_sum=(abs(logits)-logits).sum(axis=-1)/2
    union=max_sum+neg_sum+0.0001
    min_sum=torch.min(logits,groundTruth)
    intersect=min_sum.sum(axis=-1)/2+torch.abs(min_sum).sum(axis=-1)/2
    
    return  intersect/union

def span_mask_rna_tokens(
    input_ids: torch.LongTensor,
    mask_token_id: int,                # 必须传入 tokenizer.mask_token_id
    vocab_size: int = 64,
    kmer_start_id: int = 5,
    mask_prob: float = 0.2,
    max_span_length: int = 3,
    device: torch.device = None
):
    if device is None:
        device = input_ids.device

    seq_len = len(input_ids)
    special_ids = {0, 1, 2, 3, 4}      # [PAD], [UNK], [CLS], [SEP], [MASK]

    # 仅普通 k‑mer 可以 mask
    valid_positions = [pos for pos in range(seq_len) 
                       if input_ids[pos].item() not in special_ids]
    if not valid_positions:
        return input_ids.clone().to(device), torch.full_like(input_ids, -100).to(device)

    num_to_mask = max(1, int(mask_prob * len(valid_positions)))
    num_to_mask = min(num_to_mask, len(valid_positions))

    masked_positions = set()
    attempts, max_attempts = 0, num_to_mask * 10
    while len(masked_positions) < num_to_mask and attempts < max_attempts:
        start = random.choice(valid_positions)
        span_len = random.randint(1, min(max_span_length, seq_len - start))
        span = set(range(start, start + span_len))
        if span.issubset(valid_positions):
            masked_positions.update(span)
        attempts += 1

    masked_positions = list(masked_positions)[:num_to_mask]
    masked_input_ids = input_ids.clone()

    for pos in masked_positions:
        rand = random.random()
        if rand < 0.8:
            masked_input_ids[pos] = mask_token_id
        elif rand < 0.9:
            masked_input_ids[pos] = random.randint(kmer_start_id, kmer_start_id + vocab_size - 1)
        # else keep original

    labels = input_ids.clone()
    label_mask = torch.zeros(seq_len, dtype=torch.bool)
    label_mask[masked_positions] = True
    labels[~label_mask] = -100

    return masked_input_ids.to(device), labels.to(device)