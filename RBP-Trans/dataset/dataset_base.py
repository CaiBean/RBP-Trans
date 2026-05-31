from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.Seq import Seq
import torch
import h5py
from utils import *
import fm
import transformers as T


class SequenceDataset4train(torch.utils.data.Dataset):
    def __init__(self, data_total,h5_file,rbp_embedding_file,rbp_list, tokenizer, max_length=512, mode='trn'):
        df = h5py.File(h5_file)
        self.tokenizer = tokenizer
        self.max_length = max_length


        rbp_embedding = np.load(rbp_embedding_file,allow_pickle=True)
        self.embeddings = pd.Series(rbp_embedding['values'], 
                            index=rbp_embedding['index'], 
                            name=rbp_embedding['name'].item() if rbp_embedding['name'].size > 0 else None)
        self.rbp_list=rbp_list if len(rbp_list)!=0 else self.embeddings.index.tolist()

        # # self.rbp_list=np.array(['EIF3D', 'EIF3G', 'EIF3H', 'EIF4E', 'EIF4G2','EEF2'])
        # #'PRPF8','BCLAF1','GRWD1','CSTF2T','PPIG','EFTUD2','DDX3X',

        # # self.rbp_list=np.array( ['UPF1'])
        self.cellTypeDict={'HepG2':0,'K562':1,'adrenal_gland':2,'HEK293':3, 'HEK293T':4, 'Hela':5,'H9':6}



        # self.data=pd.DataFrame({'seq':df[f'{mode}_seq'],'rbp_combine_profile':np.array(df[f'{mode}_label']).tolist(),'strand':df[f'{mode}_strand'],'prefix':df[f'{mode}_prefix']})
        # # self.data_bc=pd.DataFrame({'seq':df_bc[f'{mode}_seq'],'label':df_bc[f'{mode}_label'],'prefix':df_bc[f'{mode}_prefix']})
        
        # self.data.strand=self.data.strand.str.decode(encoding='utf-8')
        # self.data['label']=1.0


        # self.data.prefix=self.data.prefix.str.decode(encoding='utf-8')
        # self.data[['rbp', 'cellType']] = self.data.prefix.str.split('_', n=1, expand=True)
        # self.data.seq=self.data.seq.str.decode(encoding='utf-8')
        # self.data.seq=self.data.seq.str.replace('T','U')
        # self.data=self.data[self.data.rbp.str.contains('|'.join(self.rbp_list))]



        self.mrna_fm, alphabet_mrna = fm.pretrained.mrna_fm_t12()
        self.batch_converter_mrna = alphabet_mrna.get_batch_converter()

        self.rna_fm, alphabet_rna = fm.pretrained.rna_fm_t12()
        self.batch_converter_rna = alphabet_rna.get_batch_converter()



        

       

        self.data=data_total[data_total['mode']==mode]
        self.data.loc[self.data['rbp'] == 'U2AF1L5,U2AF1', 'rbp'] = 'U2AF1'
        self.data=self.data[self.data.rbp.str.contains('|'.join(self.rbp_list))]
        pass



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item=self.data.iloc[index]
        rbp,cellType,ss,strand,rbp_combine_profile=item.rbp,item.cellType,item.seq,item.strand,item.rbp_combine_profile
        chr,peak_start,peak_end=item.chr,item.peak_start,item.peak_end


        rbp_combine_profile=torch.tensor(np.array(rbp_combine_profile))
        rbp_combine_profile=rbp_combine_profile if strand=='+' else rbp_combine_profile.flipud()
        rbp_combine_profile.abs_()           


        
        embedding=self.embeddings[rbp]
        
        embedding_expand=np.pad(embedding,((0,3072-embedding.shape[0]),(0,0))).astype(np.float32)
        embedding_mask=np.zeros(3072,dtype=np.bool_)
        embedding_mask[:embedding.shape[0]]=True


        seq_source=ss if strand=="+" else str(Seq(ss).reverse_complement_rna())



        data=[('RNA',seq_source)]
        batch_labels, batch_strs, batch_tokens_rna = self.batch_converter_rna(data)
        batch_tokens_rna=batch_tokens_rna.flatten()


        data_mrna=[('mRNA','M'+seq_source)]
        batch_labels, batch_strs, batch_tokens_mrna = self.batch_converter_mrna(data_mrna)
        batch_tokens_mrna=batch_tokens_mrna.flatten()


        
        
        seq_source_3mer = [seq_source[i:int(i+3)] for i in range(int(len(seq_source)-2))]# 3 mer data
        seq_character = self.tokenizer(seq_source_3mer,is_split_into_words=True, add_special_tokens= True, return_tensors='pt') ['input_ids'][0]
        seq_attention_mask=self.tokenizer(seq_source_3mer,is_split_into_words=True, add_special_tokens= True, return_tensors='pt') ['attention_mask'][0]
        seq_character_mlm,label_mlm=span_mask_rna_tokens(seq_character,mask_token_id=self.tokenizer.mask_token_id)

        return self.cellTypeDict[cellType],rbp+':'+cellType,chr,peak_start,peak_end,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,seq_character,seq_character_mlm,embedding_expand,label_mlm,rbp_combine_profile
