import pandas as pd
import numpy as np
import torch
from utils import *



class SequenceDataset4trainRank(torch.utils.data.Dataset):
    def __init__(self,df,rbp_embedding_file, tokenizer, max_length=512, mode='trn'):
        # self.data = pd.read_parquet(parquet_file)
        # self.data=self.data[self.data['mode']==mode]
        self.data=df
        self.tokenizer = tokenizer
        rbp_embedding = np.load(rbp_embedding_file,allow_pickle=True)
        
        self.embeddings = pd.Series(rbp_embedding['values'], 
                            index=rbp_embedding['index'], 
                            name=rbp_embedding['name'].item() if rbp_embedding['name'].size > 0 else None)
       
        self.cellTypeDict={'HepG2':0,'K562':1,'adrenal':2,'HEK293':3, 'HEK293T':4, 'Hela':5,'H9':6}
        self.rbp_list=self.embeddings.index.tolist()


        self.data[f'signalValue_norm'] = (df['signalValue'] - df['signalValue'].min()) / (df['signalValue'].max() - df['signalValue'].min())
        value = np.log2(df['p_value'])
        self.data[f'p_value_norm'] = (value - value.min()) / (value.max() - value.min())


        # self.rbp_embedding_expanded = []
        # self.rbp_embedding_mask = []
        # for rbp in self.rbp_list:
        #     emb = self.embeddings[rbp]          # (L, d)
        #     L, d = emb.shape
        #     padded = np.zeros((3072, d), dtype=np.float32)
        #     padded[:L, :] = emb
        #     self.rbp_embedding_expanded.append(padded)
        #     self.rbp_embedding_mask.append(np.arange(3072) < L)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample=self.data.iloc[index]
        cellType,signalValue_norm,p_value_norm,rbp_index=sample.cellType,sample.signalValue_norm,sample.p_value_norm,int(sample.rbp_index)

        seq_character,seq_attention_mask=np.array(sample.seq_character),np.array(sample.seq_attention_mask)
        seq_character=torch.tensor(seq_character,dtype=torch.int64)
        seq_character_mlm,label_mlm=span_mask_rna_tokens(seq_character)
        
        embedding=np.array(eval(self.embeddings[sample.rbp]),dtype=np.float32)
        
        
        # seq_source=np.array(list(map(lambda x:self.one_hot_dic[x],seq_source)))

        return self.cellTypeDict[cellType],seq_attention_mask,seq_character,seq_character_mlm,embedding,label_mlm,signalValue_norm,p_value_norm


