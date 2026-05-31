from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from utils import *
import transformers as T

class SequenceDataset4trainBC(torch.utils.data.Dataset):
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
        cellType,label,rbp_index=sample.cellType,int(sample.label),int(sample.rbp_index)
        seq_character,seq_attention_mask=np.array(sample.seq_character),np.array(sample.seq_attention_mask)
        seq_character=torch.tensor(seq_character,dtype=torch.int64)
        seq_character_mlm,label_mlm=span_mask_rna_tokens(seq_character,self.tokenizer.mask_token_id)
        embedding=self.embeddings[sample.RBP]
        embedding_expand=np.pad(embedding,((0,3072-embedding.shape[0]),(0,0))).astype(np.float32)
        embedding_mask=np.zeros(3072,dtype=np.bool_)
        embedding_mask[:embedding.shape[0]]=True
        batch_tokens_rna,batch_tokens_mrna=sample.tokens_rna.astype(np.int32),sample.tokens_mrna.astype(np.int32)
        sample_category=cellType+":"+str(rbp_index)
        
        
        # seq_source=np.array(list(map(lambda x:self.one_hot_dic[x],seq_source)))

        return self.cellTypeDict[cellType],sample_category,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,seq_character,seq_character_mlm,embedding_expand,label_mlm,label

