from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
from utils import *
import fm
import transformers as T



tokenizer_path = "/share/home/xuls/others/secondTongbu/ReformerSecond/Reformer/model"
h5_file = '/share/home/xuls/tongbu/Reformer/data/encode_eclip.h5'
rbp_embedding_file = '/share/home/xuls/tongbu/EncDecForRbpPositioning/data/series_with_rbp_embedding_residule_ESM2.npz'

rbp_embedding = np.load(rbp_embedding_file, allow_pickle=True)
embeddings = pd.Series(rbp_embedding['values'],
                        index=rbp_embedding['index'],
                        name=rbp_embedding['name'].item() if rbp_embedding['name'].size > 0 else None)
rbp_list = embeddings.index.tolist()
cellTypeDict = {'HepG2': 0, 'K562': 1, 'adrenal': 2, 'HEK293': 3, 'HEK293T': 4, 'Hela': 5, 'H9': 6}

mrna_fm, alphabet_mrna = fm.pretrained.mrna_fm_t12()
batch_converter_mrna = alphabet_mrna.get_batch_converter()
rna_fm, alphabet_rna = fm.pretrained.rna_fm_t12()
batch_converter_rna = alphabet_rna.get_batch_converter()

tokenizer = T.BertTokenizer.from_pretrained(tokenizer_path)
df = h5py.File(h5_file, 'r')

data_list = []
print('开始处理')

for mode in ['trn','val', 'test']:
    sequence = [ele.decode() for ele in df[f'{mode}_seq']]
    RBP, cellType = zip(*[ele.decode().split('_') for ele in df[f'{mode}_prefix']])
    profiles = np.array(df[f'{mode}_label'])
    dataTable =pd.DataFrame({'seq':df[f'{mode}_seq'],'strand':df[f'{mode}_strand'],'prefix':df[f'{mode}_prefix']})
    dataTable.prefix=dataTable.prefix.str.decode(encoding='utf-8')
    dataTable[['rbp', 'cellType']] = dataTable.prefix.str.split('_', n=1, expand=True)
    dataTable.seq=dataTable.seq.str.decode(encoding='utf-8')
    dataTable.seq=dataTable.seq.str.replace('T','U')
    dataTable['mode']=mode
    
    
    rbp_to_idx = {rbp: idx for idx, rbp in enumerate(rbp_list)}
    dataTable['rbp_index'] = dataTable['rbp'].map(rbp_to_idx)  # 修正：使用 dataTable['RBP']


    
    strands = dataTable['strand'].values
    neg_mask = (strands == '-')
    profiles[neg_mask] = profiles[neg_mask, ::-1]
    profiles = np.abs(profiles)
    dataTable['rbp_combine_profile'] = profiles.tolist()
    
    print(f'{mode}表格开始处理')
    
    seq_char_list = []
    seq_attn_mask_list = []
    tokens_rna_list = []
    tokens_mrna_list = []
    
    # 使用 tqdm 显示进度
    for idx, seq in enumerate(tqdm(sequence, desc=f'Processing {mode}')):
        # RNA tokens
        data_rna = [('RNA', seq)]
        _, _, batch_tokens_rna = batch_converter_rna(data_rna)
        tokens_rna_list.append(batch_tokens_rna.flatten())  # 1D tensor
        
        # mRNA tokens
        data_mrna = [('mRNA', 'M' + seq)]
        _, _, batch_tokens_mrna = batch_converter_mrna(data_mrna)
        tokens_mrna_list.append(batch_tokens_mrna.flatten())
        
        # 3-mer tokenization
        ss_3mer = [seq[i:i+3] for i in range(len(seq)-2)]
        encoded = tokenizer(ss_3mer, is_split_into_words=True, add_special_tokens=True, return_tensors='pt')
        seq_char_list.append(encoded['input_ids'][0])      # shape (L,)
        seq_attn_mask_list.append(encoded['attention_mask'][0])
    
    # 存储为 list of list（为了 parquet 序列化）
    dataTable['tokens_rna'] = [t.tolist() for t in tokens_rna_list]
    dataTable['tokens_mrna'] = [t.tolist() for t in tokens_mrna_list]
    dataTable['seq_character'] = [t.tolist() for t in seq_char_list]
    dataTable['seq_attention_mask'] = [t.tolist() for t in seq_attn_mask_list]
    dataTable['mode'] = mode
    
    data_list.append(dataTable)

# 合并并保存
final_df = pd.concat(data_list, axis=0, ignore_index=True)
final_df.to_parquet('eclip.parquet')  # 修正拼写
print("保存完成")