import torch 
from dataset.dataset_rank import SequenceDataset4trainRank
from model.model_rank import Model_Rank
from torch.utils.data import DataLoader
import time
import datetime
import json
import numpy as np
import random
import pandas as pd
import argparse
from torch.amp import GradScaler
import os
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import transformers as T
import  yaml
from pathlib import Path
from utils import *
from scheduler import create_scheduler
from optim import create_optimizer

def predict_ordinal_class(logits: torch.Tensor) -> torch.Tensor:
    """
    根据方案A的logits预测类别。
    logits[k] 表示 P(score ≤ k) 的 logit。
    """
    probs = torch.sigmoid(logits)          # shape: (batch, K-1)
    # 找出每行第一个 prob ≥ 0.5 的列索引，若全 <0.5 则类别为 K-1
    # 方法：用 argmax 找到第一个 True，配合一个全 False 掩码
    mask = (probs >= 0.5)                  # (batch, K-1)
    # 为每行添加一个 True 哨兵在末尾，保证 argmax 总能返回一个位置
    mask_with_sentinel = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)  # (batch, K)
    preds = mask_with_sentinel.long().argmax(dim=1)   # 第一个 True 的索引，0..K-1
    return preds

def ordinal_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        preds = predict_ordinal_class(logits)
        return (preds == labels).float().mean().item()

def validate_rank(model,df,df_path,val_dataloader,device,epoch):
    scaler = GradScaler()

    header = 'val Epoch: [{}]'.format(epoch)
    print_freq = 50 
    model.eval()


    metric_logger = MetricLogger(delimiter="  ")
  

    
    metric_logger.add_meter('loss_p_value', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_signal_value', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('acc_mlm', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('profile_p_value_max', SmoothedValue(window_size=50, fmt='{value:.6f}'))

    pred_signal_list,pred_p_list,pred_signal_list_m,pred_p_list_m=[],[],[],[]


    
    with torch.no_grad():

        for i,(cellType,batch_tokens_rna,batch_tokens_mrna,seq_attention_mask,seq_character,seq_character_mlm,rbp_embedding,embedding_mask,label_mlm,signalValue_norm,p_value_norm) in enumerate(metric_logger.log_every(val_dataloader, print_freq, header)):
            cellType,batch_tokens_rna,batch_tokens_mrna,signalValue_norm,p_value_norm,seq_attention_mask,seq_character,seq_character_mlm,label_mlm,rbp_embedding,embedding_mask=cellType.to(device),batch_tokens_rna.to(device),batch_tokens_mrna.to(device),signalValue_norm.to(device),p_value_norm.to(device)\
                ,seq_attention_mask.to(device),seq_character.to(device),seq_character_mlm.to(device),label_mlm.to(device),rbp_embedding.to(device),embedding_mask.to(device)
        
            alpha = config['alpha']*min(1,i/len(val_dataloader))  if epoch==0 else 0.4
            alpha_tensor = torch.tensor([0.0], device=device)

            with torch.autocast(device_type="cuda"):
                mlm_output,loss_signal_value,loss_p_value,loss_mlm_inner,profile_p_value_m,profile_signal_value_m,profile_p_value,profile_signal_value=model(cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,label_mlm,seq_attention_mask,seq_character,seq_character_mlm,rbp_embedding,signalValue_norm,p_value_norm,alpha_tensor)
            
            acc_mlm=sum((mlm_output.detach()[label_mlm!=-100]).argmax(-1)==label_mlm[label_mlm!=-100]).item()/len(label_mlm[label_mlm!=-100]) 

            pred_signal_list.extend(profile_signal_value.detach().cpu().tolist()) 
            pred_signal_list_m.extend(profile_signal_value_m.detach().cpu().tolist())
            pred_p_list.extend(profile_p_value.detach().cpu().tolist())
            pred_p_list_m.extend(profile_p_value_m.detach().cpu().tolist())


            metric_logger.update(acc_mlm=acc_mlm)
            metric_logger.update(loss_p_value=loss_p_value)
            metric_logger.update(loss_signal_value=loss_signal_value)
            metric_logger.update(profile_p_value_max=profile_p_value.max())
    
    df[f'pred_signal_{epoch}']=pred_signal_list
    df[f'pred_signal_{epoch}_m']=pred_signal_list_m
    df[f'pred_p_{epoch}']=pred_p_list
    df[f'pred_p_{epoch}_m']=pred_p_list_m
    df.to_parquet(df_path)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    



def train_rank(model,train_dataloader,optimizer,scheduler,warmup_steps,device,epoch):
    scaler = GradScaler()
    model.train()
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('alpha', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('acc_mlm', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_p_value', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_signal_value', SmoothedValue(window_size=50, fmt='{value:.6f}'))#pred_p_3.max()
    metric_logger.add_meter('profile_p_value_max', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('profile_signal_3_min', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50



    
    for i,(cellType,batch_tokens_rna,batch_tokens_mrna,seq_attention_mask,seq_character,seq_character_mlm,rbp_embedding,embedding_mask,label_mlm,signalValue_norm,p_value_norm) in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
        cellType,batch_tokens_rna,batch_tokens_mrna,signalValue_norm,p_value_norm,seq_attention_mask,seq_character,seq_character_mlm,label_mlm,rbp_embedding,embedding_mask=cellType.to(device),batch_tokens_rna.to(device),batch_tokens_mrna.to(device),signalValue_norm.to(device),p_value_norm.to(device)\
            ,seq_attention_mask.to(device),seq_character.to(device),seq_character_mlm.to(device),label_mlm.to(device),rbp_embedding.to(device),embedding_mask.to(device)

        optimizer.zero_grad()

        





        alpha = config['alpha']*min(1,i/len(train_dataloader))  if epoch==0 else 0.4
        alpha_tensor = torch.tensor([0.0], device=device)
        with torch.autocast(device_type="cuda"):                             
            mlm_output,loss_signal_value,loss_p_value,loss_mlm_inner,profile_p_value_m,profile_signal_value_m,profile_p_value,profile_signal_value=model(cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,label_mlm,seq_attention_mask,seq_character,seq_character_mlm,rbp_embedding,signalValue_norm,p_value_norm,alpha_tensor)
            
        loss=loss_signal_value+loss_p_value+0.1*loss_mlm_inner
        scaler.scale(loss.half()).backward()
        scaler.step(optimizer)
        scaler.update()   





        acc_mlm=sum((mlm_output.detach()[label_mlm!=-100]).argmax(-1)==label_mlm[label_mlm!=-100]).item()/len(label_mlm[label_mlm!=-100]) 


 



        metric_logger.update(alpha=alpha)
        metric_logger.update(acc_mlm=acc_mlm)
        metric_logger.update(loss_signal_value=loss_signal_value)
        metric_logger.update(loss_p_value=loss_p_value)
        metric_logger.update(profile_p_value_max=profile_p_value.max())
        metric_logger.update(profile_signal_3_min=profile_signal_value.min())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])     

        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    
    
import numpy as np
from typing import List

def calculate_gcaug_enrichment(sequences: List[str], motif: str = "GCATG") -> dict:
    """
    计算500条长度200bp的RNA序列中GCAUG motif的富集倍数。

    Parameters:
    -----------
    sequences : List[str]
        包含500个RNA序列的列表，每个序列由'A','C','G','U'组成，长度应为200。
    motif : str
        要搜索的motif，默认为'GCAUG'。

    Returns:
    --------
    dict
        包含以下字段：
        - observed_frac: 观测到的含motif序列比例
        - bg_frac_uniform: 基于均匀背景(25%每碱基)的期望概率
        - bg_frac_actual: 基于输入序列实际单碱基频率的期望概率
        - enrichment_uniform: 相对于均匀背景的富集倍数
        - enrichment_actual: 相对于实际背景的富集倍数
        - motif_counts: 每条序列是否含motif的布尔列表（用于下游分析）
    """

    
    # 1. 观测比例
    motif_present = [motif in seq for seq in sequences]
    observed_frac = np.mean(motif_present)
    
    # 2. 计算实际碱基频率（用于精确背景）
    all_seq = "".join(sequences)
    total_bases = len(all_seq)
    freq = {base: all_seq.count(base) / total_bases for base in "ACGT"}
    
    # 3. 基于均匀假设的背景概率
    p_uniform = (0.25 ** len(motif))
    # 长度为200的序列中，motif可能起始位点数
    L = 200
    possible_starts = L - len(motif) + 1
    # 至少出现一次的概率 (泊松近似)
    lambda_uniform = possible_starts * p_uniform
    bg_frac_uniform = 1 - np.exp(-lambda_uniform)


    # 4. 基于实际碱基频率的背景概率
    p_actual = 1.0
    for base in motif:
        p_actual *= freq[base]
    lambda_actual = possible_starts * p_actual
    bg_frac_actual = 1 - np.exp(-lambda_actual)
    
    # 5. 富集倍数
    enrichment_uniform = observed_frac / bg_frac_uniform if bg_frac_uniform > 0 else np.inf
    enrichment_actual = observed_frac / bg_frac_actual if bg_frac_actual > 0 else np.inf
    
    return {
        "observed_frac": observed_frac,
        "bg_frac_uniform": bg_frac_uniform,
        "bg_frac_actual": bg_frac_actual,
        "enrichment_uniform": enrichment_uniform,
        "enrichment_actual": enrichment_actual,
        "motif_counts": motif_present
    }

# 使用示例：
# sequences = ["ACGU...", ...]  你的500条序列
# result = calculate_gcaug_enrichment(sequences)
# print(f"富集倍数 (实际背景): {result['enrichment_actual']:.3f}")

def main(args, config):

    tokenizer_path="/share/home/xuls/others/secondTongbu/ReformerSecond/Reformer/model"
    device=f'{args.cuda}' if torch.cuda.is_available() else 'cpu'   # 设备
    print(f'device:{device}')
    tokenizer = T.BertTokenizer.from_pretrained(tokenizer_path)
    warmup_steps = config['schedular']['warmup_epochs']    
    start_epoch=6

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    import pandas as pd
    from sklearn.model_selection import train_test_split # 导入 train_test_split 函数
    df_path='/share/home/xuls/tongbu/EncDecForRbpPositioning/Rank_HepG2_RBFOX2_rep1.parquet'
    df_1=pd.read_parquet(df_path)
    import pandas as pd

    # df 是你原有的 DataFrame

    rename_dict = {}
    for col in df_1.columns:
        if col == 'signalValue':
            rename_dict[col] = 'p_value'
        elif col == 'p_value':
            rename_dict[col] = 'signalValue'
        elif col == 'signalValue_norm':
            rename_dict[col] = 'p_value_norm'
        elif col == 'p_value_norm':
            rename_dict[col] = 'signalValue_norm'
        elif col.startswith('pred_signal_'):
            # 保留后面的数字和可能存在的 "_m" 后缀
            suffix = col[len('pred_signal_'):]  # 例如 '10' 或 '10_m'
            rename_dict[col] = 'pred_p_' + suffix
        elif col.startswith('pred_p_'):
            suffix = col[len('pred_p_'):]
            rename_dict[col] = 'pred_signal_' + suffix

    df_1.rename(columns=rename_dict, inplace=True)
    df_1.to_parquet('/share/home/xuls/tongbu/EncDecForRbpPositioning/Rank_HepG2_RBFOX2_rep2_exchangeColumnOrder.parquet')
    
    from tqdm import tqdm
    df_1['motif']=df_1.apply(lambda row:'GCATG' in row.seq[256-100:256+200] if row.strand=='+' else 'CATGC' in row.seq[256-100:256+100],axis=1)
    pass


    df_1['bed_text_original']=df_1.apply(lambda row:row['chrome']+'\t'+str(row.peak_start)+'\t'+str(row.peak_end)+'\t'+row['name']+'\t'+str(row.score)+'\t'+row.strand+'\t'+str(row.signalValue)+'\t'+str(row.p_value)+'\t-1\t-1',axis=1)
    # df_original_rep1=df.sort_values('signalValue')
    # with open('RBFOX2_original_rank_rep1.bed','w')as f:
    #     [print(ele,file=f) for ele in df_original_rep1.bed_text_original.tolist()]

    df_1['bed_text_norm']=df_1.apply(lambda row:row['chrome']+'\t'+str(row.peak_start)+'\t'+str(row.peak_end)+'\t'+row['name']+'\t'+str(row.score)+'\t'+row.strand+'\t'+str(row.signalValue_norm)+'\t'+str(row.p_value_norm)+'\t-1\t-1',axis=1)
    # df_norm_rep1=df.sort_values('signalValue_norm')
    # with open('RBFOX2_original_rank_norm_rep1.bed','w')as f:
    #     [print(ele,file=f) for ele in df_norm_rep1.bed_text_norm.tolist()]

    df_1['bed_text_mine']=df_1.apply(lambda row:row['chrome']+'\t'+str(row.peak_start)+'\t'+str(row.peak_end)+'\t'+row['name']+'\t'+str(row.score)+'\t'+row.strand+'\t'+str(row.pred_signal_11_m*0.2+0.8*row.signalValue_norm)+'\t'+str(row.pred_p_11_m*0.2+0.8*row.p_value_norm)+
                                     '\t-1\t-1',axis=1)
    # df_mine_rep1=df.sort_values(f'pred_signal_8_m')
    # with open(f'RBFOX2_signalValue_rank_all_rep1.bed','w')as f:
    #     [print(ele,file=f) for ele in df_mine_rep1.bed_text_mine.tolist()]
    


    df_original_7000_rep1=df_1.sort_values('signalValue').iloc[-7000:,:]
    with open('RBFOX2_original_rank_7000_rep1.bed','w')as f:
        [print(ele,file=f) for ele in df_original_7000_rep1.bed_text_original.tolist()]

    df_norm_7000_rep1=df_1.sort_values('signalValue_norm').iloc[-7000:,:]
    with open('RBFOX2_original_rank_norm_7000_rep1.bed','w')as f:
        [print(ele,file=f) for ele in df_norm_7000_rep1.bed_text_norm.tolist()]

    df_mine_7000_rep1=df_1.sort_values(f'pred_signal_8_m').iloc[-7000:,:]
    with open(f'RBFOX2_signalValue_rank_all_7000_rep1.bed','w')as f:
        [print(ele,file=f) for ele in df_mine_7000_rep1.bed_text_mine.tolist()]


    df_original_1000_rep1=df_1.sort_values('signalValue').iloc[-30000:,:]
    with open('RBFOX2_original_rank_1000_rep1.bed','w')as f:
        [print(ele,file=f) for ele in df_original_1000_rep1.bed_text_original.tolist()]

    df_norm_1000_rep1=df_1.sort_values('signalValue_norm').iloc[-30000:,:]
    with open('RBFOX2_original_rank_norm_1000_rep1.bed','w')as f:
        [print(ele,file=f) for ele in df_norm_1000_rep1.bed_text_norm.tolist()]

    df_mine_1000_rep1=df_1.sort_values(f'pred_signal_11_m').iloc[-30000:,:]
    with open(f'RBFOX2_signalValue_rank_all_1000_rep1.bed','w')as f:
        [print(ele,file=f) for ele in df_mine_1000_rep1.bed_text_mine.tolist()]


    df_path='/share/home/xuls/tongbu/EncDecForRbpPositioning/Rank_HepG2_RBFOX2_rep2.parquet'
    df_2=pd.read_parquet(df_path)
    import pandas as pd

    # df 是你原有的 DataFrame

    rename_dict = {}
    for col in df_2.columns:
        if col == 'signalValue':
            rename_dict[col] = 'p_value'
        elif col == 'p_value':
            rename_dict[col] = 'signalValue'
        elif col == 'signalValue_norm':
            rename_dict[col] = 'p_value_norm'
        elif col == 'p_value_norm':
            rename_dict[col] = 'signalValue_norm'
        elif col.startswith('pred_signal_'):
            # 保留后面的数字和可能存在的 "_m" 后缀
            suffix = col[len('pred_signal_'):]  # 例如 '10' 或 '10_m'
            rename_dict[col] = 'pred_p_' + suffix
        elif col.startswith('pred_p_'):
            suffix = col[len('pred_p_'):]
            rename_dict[col] = 'pred_signal_' + suffix

    df_2.rename(columns=rename_dict, inplace=True)
    df_2.to_parquet('/share/home/xuls/tongbu/EncDecForRbpPositioning/Rank_HepG2_RBFOX2_rep2_exchangeColumnOrder.parquet')
    
    from tqdm import tqdm
    df_2['motif']=df_2.apply(lambda row:'GCATG' in row.seq[256-100:256+200] if row.strand=='+' else 'CATGC' in row.seq[256-100:256+100],axis=1)


    df_2['bed_text_original']=df_2.apply(lambda row:row['chrome']+'\t'+str(row.peak_start)+'\t'+str(row.peak_end)+'\t'+row['name']+'\t'+str(row.score)+'\t'+row.strand+'\t'+str(row.signalValue)+'\t'+str(row.p_value)+'\t-1\t-1',axis=1)
    # df_original_rep2=df.sort_values('signalValue')
    # with open('RBFOX2_original_rank_rep2.bed','w')as f:
    #     [print(ele,file=f) for ele in df_original_rep2.bed_text_original.tolist()]

    df_2['bed_text_norm']=df_2.apply(lambda row:row['chrome']+'\t'+str(row.peak_start)+'\t'+str(row.peak_end)+'\t'+row['name']+'\t'+str(row.score)+'\t'+row.strand+'\t'+str(row.signalValue_norm)+'\t'+str(row.p_value_norm)+'\t-1\t-1',axis=1)
    # df_norm_rep2=df.sort_values('signalValue_norm')
    # with open('RBFOX2_original_rank_norm_rep2.bed','w')as f:
    #     [print(ele,file=f) for ele in df_norm_rep2.bed_text_norm.tolist()]

    df_2['bed_text_mine']=df_2.apply(lambda row:row['chrome']+'\t'+str(row.peak_start)+'\t'+str(row.peak_end)+'\t'+row['name']+'\t'+str(row.score)+'\t'+row.strand+'\t'+str(row.pred_signal_9_m*0.2+0.8*row.signalValue_norm)+'\t'+str(row.pred_p_9_m*0.2+row.p_value_norm*0.8)
                                    +'\t-1\t-1',axis=1)
    # df_mine_rep2=df.sort_values(f'pred_signal_9_m')
    # with open(f'RBFOX2_signalValue_rank_all_rep2.bed','w')as f:
    #     [print(ele,file=f) for ele in df_mine_rep2.bed_text_mine.tolist()]
    


    df_original_7000_rep2=df_2.sort_values('signalValue').iloc[-7000:,:]
    with open('RBFOX2_original_rank_7000_rep2.bed','w')as f:
        [print(ele,file=f) for ele in df_original_7000_rep2.bed_text_original.tolist()]

    df_norm_7000_rep2=df_2.sort_values('signalValue_norm').iloc[-7000:,:]
    with open('RBFOX2_original_rank_norm_7000_rep2.bed','w')as f:
        [print(ele,file=f) for ele in df_norm_7000_rep2.bed_text_norm.tolist()]

    df_mine_7000_rep2=df_2.sort_values(f'pred_signal_9_m').iloc[-7000:,:]
    with open(f'RBFOX2_signalValue_rank_all_7000_rep2.bed','w')as f:
        [print(ele,file=f) for ele in df_mine_7000_rep2.bed_text_mine.tolist()]


    df_original_1000_rep2=df_2.sort_values('signalValue').iloc[-30000:,:]
    with open('RBFOX2_original_rank_1000_rep2.bed','w')as f:
        [print(ele,file=f) for ele in df_original_1000_rep2.bed_text_original.tolist()]

    df_norm_1000_rep2=df_2.sort_values('signalValue_norm').iloc[-30000:,:]
    with open('RBFOX2_original_rank_norm_1000_rep2.bed','w')as f:
        [print(ele,file=f) for ele in df_norm_1000_rep2.bed_text_norm.tolist()]

    df_mine_1000_rep2=df_2.sort_values(f'pred_signal_9_m').iloc[-30000:,:]
    with open(f'RBFOX2_signalValue_rank_all_1000_rep2.bed','w')as f:
        [print(ele,file=f) for ele in df_mine_1000_rep2.bed_text_mine.tolist()]


    table_mine=pd.read_csv('/share/home/xuls/bioinformaticsXls/bioinformaticsAnalysis/idrAnalysis/sample-idr-rbfox2_mine_1000',delimiter='\t')
    table_original=pd.read_csv('/share/home/xuls/bioinformaticsXls/bioinformaticsAnalysis/idrAnalysis/sample-idr-rbfox2_1000',delimiter='\t')
    table_mine.columns=['chrome','chromStart','chromEnd','name','score','strand','signalValue','p_value','q_value','summit','localIDR','globalIDR','rep1_chromStart','rep1_chromEnd','rep1_signalValue','rep1_summit','rep2_chromStart','rep2_chromEnd','rep2_signalValue','rep2_summit']
    table_mine.columns=['chrome','chromStart_idr','chromEnd_idr','name','score','strand','signalValue_idr','p_value_idr','q_value_idr','summit','localIDR','globalIDR','rep1_chromStart','rep1_chromEnd','rep1_signalValue','rep1_summit','rep2_chromStart','rep2_chromEnd','rep2_signalValue','rep2_summit']
    table_original.columns=['chrome','chromStart_idr','chromEnd_idr','name','score','strand','signalValue_idr','p_value_idr','q_value_idr','summit','localIDR','globalIDR','rep1_chromStart','rep1_chromEnd','rep1_signalValue','rep1_summit','rep2_chromStart','rep2_chromEnd','rep2_signalValue','rep2_summit']

    table_mine['rep1_index']=table_mine.apply(lambda row:f'{row[12]}_{row[13]}',axis=1)
    table_mine['rep2_index']=table_mine.apply(lambda row:f'{row[16]}_{row[17]}',axis=1)
    
    df_mine_1000_rep1['rep1_index']=df_mine_1000_rep1.apply(lambda row:f'{row.peak_start}_{row.peak_end}',axis=1)
    df_mine_1000_rep2['rep2_index']=df_mine_1000_rep2.apply(lambda row:f'{row.peak_start}_{row.peak_end}',axis=1)

    _=pd.merge(left=table_mine,right=df_mine_1000_rep1,on='rep1_index')
    table_mine_total=pd.merge(left=_,right=df_mine_1000_rep2,on='rep2_index')
    table_mine_total['p_value_geometric_mean']=np.sqrt(table_mine_total.p_value_x*table_mine_total.p_value_y)
    table_mine_total['signalValue_geometric_mean']=np.sqrt(table_mine_total.signalValue_x*table_mine_total.signalValue_y)
    # df.iloc[-27].seq.index('CATGC')
    # seq_top500_1=df.sort_values('signalValue').iloc[-7000:,:].seq.str[256-100:256+100].tolist()
    # result1=calculate_gcaug_enrichment(seq_top500_1)

    # seq_top500_2=df.sort_values('pred_p_11_m').iloc[-100:,:].seq.str[256-100:256+100].tolist()
    # result2=calculate_gcaug_enrichment(seq_top500_2)
    # pass
    
    # rbp_list_42=sorted(list(set(df.RBP)))
    # train_df,test_df=df[~df.RBP.str.contains("|".join(RBP_list[2::5]))],df[df.RBP.str.contains("|".join(RBP_list[2::5]))]
    # _, alphabet_mrna = fm.pretrained.rna_fm_t12()
    # batch_converter_rna = alphabet_mrna.get_batch_converter()

    # _, alphabet_rna = fm.pretrained.mrna_fm_t12()
    # batch_converter_mrna = alphabet_rna.get_batch_converter()
    # rbp_to_idx_42 = {rbp: idx for idx, rbp in enumerate(rbp_list_42)}
    # df['rbp_index'] = df['RBP'].map(rbp_to_idx_42)
    # seq_char_list = []
    # seq_attn_mask_list = []
    # tokens_rna_list = []
    # tokens_mrna_list = []
    # for i,seq in tqdm(enumerate(df['seq'])):
    #     seq=seq.replace('T','U')
    #     data = [('RNA', seq)]
    #     _, _, batch_tokens_rna = batch_converter_rna(data)
    #     tokens_rna_list.append(batch_tokens_rna.flatten())  # 1D tensor

    #     data_mrna = [('mRNA', 'M' + seq)]
    #     _, _, batch_tokens_mrna = batch_converter_mrna(data_mrna)
    #     tokens_mrna_list.append(batch_tokens_mrna.flatten())

        # ss_3mer = [seq[i:i+3] for i in range(len(seq)-2)]
        # encoded = tokenizer(ss_3mer, is_split_into_words=True, add_special_tokens=True, return_tensors='pt')
        # seq_char_list.append(encoded['input_ids'][0])      # shape (L,)
        # seq_attn_mask_list.append(encoded['attention_mask'][0])


    # df['tokens_rna'] = [ele.numpy().astype(np.int32) for ele in tokens_rna_list]
    # df['tokens_mrna'] = [ele.numpy().astype(np.int32) for ele in tokens_mrna_list]
    # df['seq_character'] = [ele.numpy().astype(np.int32) for ele in seq_char_list]
    # df['seq_attention_mask'] =[  ele.numpy().astype(np.int32)  for ele in seq_attn_mask_list]
    # df.to_parquet('/share/home/xuls/tongbu/EncDecForRbpPositioning/Rank_HepG2_RBFOX2_rep1.parquet')
    # print('OK')
    


    print("Creating model")
    model = Model_Rank(tokenizer, config,device)
    model = model.to(device)

    arg_opt = AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)  # 先创建 optimizer（绑定原始模型参数）
    arg_sche = AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    start_epoch = 6
    if args.checkpoint:
        if args.resume:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        # 加载模型权重到原始模型
        checkpoint = torch.load(args.checkpoint, map_location='cpu')['model']
        keys_list = list(checkpoint.keys())
        for key in keys_list:
            if 'orig_mod.' in key:
                deal_key = key.replace('_orig_mod.', '')
                checkpoint[deal_key] = checkpoint[key]
                del checkpoint[key]
        
        model.load_state_dict(checkpoint,strict=False)
        print('load checkpoint from %s' % args.checkpoint)

    # 编译模型（放在最后，所有准备就绪）
    # model = torch.compile(model)


    '''
        训练模型
    '''
    print("Creating dataset")

    val_dataset= SequenceDataset4trainRank(args.parquet_path,df,config['rbp_embedding'],tokenizer,max_length=512,mode='val')
    train_dataset =val_dataset
    train_dataset=SequenceDataset4trainRank(args.parquet_path,df,config['rbp_embedding'],tokenizer,max_length=512,mode='trn')

   
    


    model_without_ddp = model
    if args.distributed != False:
        model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)
        model_without_ddp = model.module 
        
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True, num_workers=20,
                                pin_memory=True,sampler=train_sampler)  # TODO: Check whether drop_last=True?e)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True, num_workers=20,
                                pin_memory=True,sampler=val_sampler)  # TODO: Check whether drop_last=True?e)
    else:
        train_dataloader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,num_workers=16,persistent_workers=True,drop_last=False,pin_memory=True)    # 数据加载器
        val_dataloader=DataLoader(val_dataset,batch_size=config['batch_size'],shuffle=False,num_workers=16,persistent_workers=True,drop_last=False,pin_memory=True) 

    


    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config['epochs']):
        print(f"/share/home/xuls/tongbu/mnist-clip-main/params/rank/rep2/checkpoint_0{epoch}.pth")
        checkpoint = torch.load(f"/share/home/xuls/tongbu/mnist-clip-main/params/rank/rep2/checkpoint_0{epoch}.pth", map_location='cpu')['model']
        keys_list = list(checkpoint.keys())
        for key in keys_list:
            if 'orig_mod.' in key:
                deal_key = key.replace('_orig_mod.', '')
                checkpoint[deal_key] = checkpoint[key]
                del checkpoint[key]
        
        model.load_state_dict(checkpoint)
        # if epoch>0:
        #     lr_scheduler.step(epoch+warmup_steps)  
        # # val_stats=validate_rank(model,df,df_path,val_dataloader,device,epoch)
        # train_stats=train_rank(model,train_dataloader,optimizer,lr_scheduler,warmup_steps,device,epoch)
        
        # if is_main_process():  
        #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                     'epoch': epoch,
        #                 }                     
        #     save_obj = {
        #         'model': model_without_ddp.state_dict(),  # 保存原始模型
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'config':config,
        #         'epoch': epoch,
        #     }
        #     torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
        #     print('save checkpoint_%02d.pth successfully'%epoch )
        #     with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

            

            
        val_stats=validate_rank(model,df,df_path,val_dataloader,device,epoch)
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                    'epoch': epoch,
                }  
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")   
        print(val_stats)

    # dist.barrier()  
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print('Training time {}'.format(total_time_str)) 
    

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--distributed',default=False,action='store_true')
    parser.add_argument('-c','--cuda',default='cuda:1')
    parser.add_argument('--config', default='./configs/Pretrain_rank.yaml')
    parser.add_argument('--output_dir', default='params/rank/rep1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', default="") #/share/home/xuls/tongbu/mnist-clip-main/params/rank/rep1/checkpoint_10.pth
    parser.add_argument('--resume', default=True, type=bool)
    parser.add_argument('--parquet_path', help='input parquet file',default='/share/home/xuls/tongbu/EncDecForRbpPositioning/DataProcess/rank_HepG2_RBFOX2_rep1.parquet')#/share/home/xuls/tongbu/mnist-clip-main/data/Reformer_bc.parquet




    args = parser.parse_args()
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    else:
        local_rank=None
   
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))  
    main(args, config)