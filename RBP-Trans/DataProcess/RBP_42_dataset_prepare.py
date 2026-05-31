import fm
import pandas as pd
from tqdm import tqdm


df=pd.read_parquet('/share/home/xuls/tongbu/MsipNet/RBP42Dataset.parquet')

rbp_list_42=sorted(list(set(df.RBP)))
# train_df,test_df=df[~df.RBP.str.contains("|".join(RBP_list[2::5]))],df[df.RBP.str.contains("|".join(RBP_list[2::5]))]
_, alphabet_mrna = fm.pretrained.rna_fm_t12()
batch_converter_rna = alphabet_mrna.get_batch_converter()

_, alphabet_rna = fm.pretrained.mrna_fm_t12()
batch_converter_mrna = alphabet_rna.get_batch_converter()
rbp_to_idx_42 = {rbp: idx for idx, rbp in enumerate(rbp_list_42)}
df['rbp_index'] = df['RBP'].map(rbp_to_idx_42)
seq_char_list = []
seq_attn_mask_list = []
tokens_rna_list = []
tokens_mrna_list = []
for i,seq in tqdm(enumerate(df['seq'])):
    data = [('RNA', seq)]
    _, _, batch_tokens_rna = batch_converter_rna(data)
    tokens_rna_list.append(batch_tokens_rna.flatten())  # 1D tensor

    data_mrna = [('mRNA', 'M' + seq)]
    _, _, batch_tokens_mrna = batch_converter_mrna(data_mrna)
    tokens_mrna_list.append(batch_tokens_mrna.flatten())

    ss_3mer = [seq[i:i+3] for i in range(len(seq)-2)]
    encoded = tokenizer(ss_3mer, is_split_into_words=True, add_special_tokens=True, return_tensors='pt')
    seq_char_list.append(encoded['input_ids'][0])      # shape (L,)
    seq_attn_mask_list.append(encoded['attention_mask'][0])


df['tokens_rna'] = [ele.numpy() for ele in tokens_rna_list]
df['tokens_mrna'] = [ele.numpy() for ele in tokens_mrna_list]
df['seq_character'] = [ele.numpy() for ele in seq_char_list]
df['seq_attention_mask'] =[  ele.numpy()  for ele in seq_attn_mask_list]
df.to_parquet('/share/home/xuls/tongbu/MsipNet/RBP42Dataset.parquet')
