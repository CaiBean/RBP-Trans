import pandas as pd
from pyfaidx import Fasta
from Bio import Seq
import numpy as np
from tqdm   import tqdm
import os
import transformers as T
import fm
from bisect import bisect_right

def build_strand_specific_interval_index(df, rbp_list):
    """
    为每个 RBP 构建按 (染色体, 链) 分组的区间索引。
    
    参数:
        df: 包含所有 peak 的 DataFrame，必须包含列 'rbp', 'chr', 'peak_start', 'peak_end', 'strand'
        rbp_list: 固定顺序的 RBP 名称列表
    
    返回:
        rbp_indices: dict, key 为 RBP 名，
                     value 为 dict: key 为 (chrom, strand) 元组，
                                   value 为按 start 排序的列表，元素为 (start, end, chrom, orig_start, orig_end, strand)
    """
    rbp_indices = {}
    for rbp in rbp_list:
        rbp_df = df[df['rbp'] == rbp]
        key_dict = {}
        for (chrom, strand), group in rbp_df.groupby(['chr', 'strand']):
            intervals = [(row['peak_start'], row['peak_end'], chrom, row['peak_start'], row['peak_end'], strand)
                         for _, row in group.iterrows()]
            intervals.sort(key=lambda x: x[0])  # 按 start 排序
            key_dict[(chrom, strand)] = intervals
        rbp_indices[rbp] = key_dict
    return rbp_indices

def find_first_overlap_strand_specific(chrom, start, end, strand, rbp_index):
    """
    在指定 RBP 的索引中查找与查询区间 [start, end] 且链相同的重叠区间。
    
    返回:
        (overlap_bool, detail_string)
        overlap_bool: True/False
        detail_string: 若重叠，返回形如 'chr7:102221500-102221513(+)' 的字符串，否则为 None
    """
    key = (chrom, strand)
    if key not in rbp_index:
        return False, None
    intervals = rbp_index[key]
    if not intervals:
        return False, None

    # 二分查找第一个 start > end 的位置
    idx = bisect_right(intervals, (end, float('inf'), '', 0, 0, ''))
    if idx > 0:
        s, e, c, orig_s, orig_e, strand_found = intervals[idx - 1]
        if e >= start:
            # 找到一个重叠区间
            return True, f"{c}:{orig_s}-{orig_e}({strand_found})"
    return False, None

def add_strand_specific_labels(df, rbp_order):
    """
    为合并后的 DataFrame 添加 'label' 和 'label_detail' 列（考虑链特异性）。
    
    参数:
        df: 包含所有 peak 的 DataFrame，列需包含 'rbp', 'chr', 'peak_start', 'peak_end', 'strand'
        rbp_order: 固定顺序的 RBP 列表，例如 ['RBFOX2', 'PRPF8', ...]
    
    返回:
        df_copy: 添加了 'label' 和 'label_detail' 列的 DataFrame 副本
    """
    # 构建全局索引（一次性）
    rbp_indices = build_strand_specific_interval_index(df, rbp_order)

    labels = []
    details = []
    for _, row in tqdm(df.iterrows()):
        chrom = row['chr']
        start = row['peak_start']
        end = row['peak_end']
        strand = row['strand']
        own_rbp = row['rbp']

        label = []
        detail = []
        for rbp in rbp_order:
            if rbp == own_rbp:
                # 自身必然重叠
                label.append(1)
                detail.append(f"{chrom}:{start}-{end}({strand})")
            else:
                overlap, info = find_first_overlap_strand_specific(chrom, start, end, strand, rbp_indices[rbp])
                label.append(1 if overlap else 0)
                detail.append(info)
        labels.append(label)
        details.append(detail)

    df_copy = df.copy()
    df_copy['label'] = labels
    df_copy['label_detail'] = details
    return df_copy



if __name__=='__main__':
    dataset_path='/share/home/xuls/dataset/ABC_CLIP_RBFOX2_GSM6561051/'
    fasta_path="/share/home/xuls/bioinformaticsXls/bioinformaticsFiles/GRCh38.primary_assembly.genome.fa"
    tokenizer_path = "/share/home/xuls/others/secondTongbu/ReformerSecond/Reformer/model"
    rbp_order = ['RBFOX2', 'PRPF8', 'PUM2', 'IGF2BP2', 'FAM120A', 'ZC3H11A', 'EIF3G', 'LIN28B', 'DDX3', 'SF3B4']
    mrna_fm, alphabet_mrna = fm.pretrained.mrna_fm_t12()
    batch_converter_mrna = alphabet_mrna.get_batch_converter()
    rna_fm, alphabet_rna = fm.pretrained.rna_fm_t12()
    batch_converter_rna = alphabet_rna.get_batch_converter()

    tokenizer = T.BertTokenizer.from_pretrained(tokenizer_path)
    genome = Fasta(fasta_path, as_raw=True)
    table_list=[]
    file_paths=os.listdir(dataset_path)
    for file_path in file_paths:
        table=pd.read_csv(dataset_path+file_path,delimiter='\t')
        table.columns=['chr','peak_start','peak_end','signalValue','p_value','strand']
        datasetID,_,_,cellType,rep,rbp=file_path.split(".")[0].split('_')
        table['name']=file_path.split('.')[0]
        table['cellType']=cellType
        table['rbp']=rbp
        table_list.append(table)
    RBP_10_table=pd.concat(table_list)
    RBP_10_table=RBP_10_table[RBP_10_table.chr.isin([f'chr{i}' for i in range(1,23)]+['chrX','chrY'])]
    RBP_10_table['seq_start']=(RBP_10_table['peak_end']+RBP_10_table['peak_start'])//2-255

    RBP_10_table['seq_end']=(RBP_10_table['peak_end']+RBP_10_table['peak_start'])//2+256

    print(f'开始收集序列')
    seq_list=[]
    for i in tqdm(range(len(RBP_10_table))):
        row=RBP_10_table.iloc[i]
        strand=row.strand
        seq=genome[row.chr][int(row.seq_start):int(row.seq_end+1)].upper()
        seq_list.append(seq if strand=='+' else str(Seq.Seq(seq).reverse_complement()))

    RBP_10_table['seq']=np.array(seq_list)


    mode_list=[np.random.choice(['trn', 'val','test'], p=[0.95, 0.04,0.01]) for i in range(len(RBP_10_table))]
    RBP_10_table['mode']=mode_list




    data_list = []
    print('开始处理')

    for mode in ['trn', 'val','test']:
        RBP_10_table_part=RBP_10_table[RBP_10_table['mode']==mode]
        
        seq_char_list = []
        seq_attn_mask_list = []
        tokens_rna_list = []
        tokens_mrna_list = []
        
        # 使用 tqdm 显示进度
        for idx, seq in enumerate(tqdm(RBP_10_table_part.seq, desc=f'Processing {mode}')):
            # RNA tokens
            seq=seq.replace('T','U')
            data_rna = [('RNA', seq)]
            _, _, batch_tokens_rna = batch_converter_rna(data_rna)
            tokens_rna_list.append(np.array(batch_tokens_rna.flatten(),dtype=np.int8))  # 1D tensor
            
            # mRNA tokens
            data_mrna = [('mRNA', 'M' + seq)]
            _, _, batch_tokens_mrna = batch_converter_mrna(data_mrna)
            tokens_mrna_list.append(np.array(batch_tokens_mrna.flatten(),dtype=np.int8))  # 1D tensor
            
            # 3-mer tokenization
            ss_3mer = [seq[i:i+3] for i in range(len(seq)-2)]
            encoded = tokenizer(ss_3mer, is_split_into_words=True, add_special_tokens=True, return_tensors='pt')
            seq_char_list.append(np.array(encoded['input_ids'][0],dtype=np.int8))      # shape (L,)
            seq_attn_mask_list.append(np.array(encoded['attention_mask'][0],dtype=np.int8))
        
        # 存储为 list of list（为了 parquet 序列化）
        RBP_10_table_part.loc[:,'tokens_rna'] =  tokens_rna_list
        RBP_10_table_part.loc[:,'tokens_mrna'] = tokens_mrna_list
        RBP_10_table_part.loc[:,'seq_character'] = seq_char_list
        RBP_10_table_part.loc[:,'seq_attention_mask'] =seq_attn_mask_list
    
        
        data_list.append(RBP_10_table_part)
    peak_df = pd.concat(data_list, axis=0, ignore_index=True)
    print('开始寻找全局共结合关系')
    peak_df_withLabel_detail=add_strand_specific_labels(peak_df,rbp_order)
    peak_df_withLabel_detail.to_parquet(f'ABC_{cellType}_{datasetID}.parquet')  # 修正拼写
    print("保存完成")