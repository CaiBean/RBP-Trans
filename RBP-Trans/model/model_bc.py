from layers.layers import *
from layers.xbert import *
import fm




# class ProteinCompressor(nn.Module):
#     def __init__(self, d_model, n_heads, n_queries=512, dropout=0.1):
#         super().__init__()
#         assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.head_dim = d_model // n_heads
#         self.n_queries = n_queries

#         # 可学习的查询向量 (1, n_queries, d_model)
#         self.query = nn.Parameter(torch.randn(1, n_queries, d_model))

#         # 多头投影矩阵（通过线性层实现，但输出维度保持 d_model，内部自动分头）
#         self.W_q = nn.Linear(d_model, d_model, bias=False)
#         self.W_k = nn.Linear(d_model, d_model, bias=False)
#         self.W_v = nn.Linear(d_model, d_model, bias=False)
#         self.out_proj = nn.Linear(d_model, d_model, bias=False)

#         self.dropout = nn.Dropout(dropout)
#         self.scale = self.head_dim ** 0.5

#     def forward(self, protein_emb, protein_mask):
#         """
#         protein_emb: (B, L_p, D)
#         protein_mask: (B, L_p)  # 1 for valid, 0 for padding
#         return: (B, n_queries, D)
#         """
#         B = protein_emb.size(0)

#         # 1. 准备 Q, K, V (保持原始维度，后续reshape分头)
#         queries = self.query.expand(B, -1, -1)          # (B, n_queries, D)
#         Q = self.W_q(queries)                           # (B, n_queries, D)
#         K = self.W_k(protein_emb)                       # (B, L_p, D)
#         V = self.W_v(protein_emb)                       # (B, L_p, D)

#         # 2. 分头: (B, n_heads, seq_len, head_dim)
#         def reshape_for_heads(x, seq_len):
#             return x.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

#         Q = reshape_for_heads(Q, self.n_queries)        # (B, n_heads, n_queries, head_dim)
#         K = reshape_for_heads(K, protein_emb.size(1))   # (B, n_heads, L_p, head_dim)
#         V = reshape_for_heads(V, protein_emb.size(1))   # (B, n_heads, L_p, head_dim)

#         # 3. 计算注意力分数（每个头独立）
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, n_heads, n_queries, L_p)

#         # 4. 应用蛋白掩码（扩展维度以匹配多头）
#         # protein_mask: (B, L_p) -> (B, 1, 1, L_p) 以便广播
#         attn_scores = attn_scores.masked_fill(~protein_mask[:, None, None, :], float('-inf'))

#         # 5. Softmax + Dropout
#         attn_probs = F.softmax(attn_scores, dim=-1)     # (B, n_heads, n_queries, L_p)
#         attn_probs = self.dropout(attn_probs)

#         # 6. 加权求和得到每个头的输出
#         head_output = torch.matmul(attn_probs, V)        # (B, n_heads, n_queries, head_dim)

#         # 7. 合并头
#         concat_output = head_output.transpose(1, 2).contiguous().view(B, self.n_queries, self.d_model)  # (B, n_queries, D)

#         # 8. 最终输出投影
#         output = self.out_proj(concat_output)            # (B, n_queries, D)
#         return output
    


# class RBP_encoder(nn.Module):
#     def __init__(self,hidden_dim,cellTypeDim,num_head,rbp_feature_length,cell_type_related,dropout):
#         super().__init__()
#         self.cell_embedding=nn.Embedding(cell_type_related,cellTypeDim)
#         self.cell_gamma_proj=nn.Linear(cellTypeDim,hidden_dim)
#         self.cell_beta_proj=nn.Linear(cellTypeDim,hidden_dim)
#         self.dense1=nn.Linear(in_features=1280,out_features=2048)
#         self.dense2=nn.Linear(in_features=2048,out_features=1024)
#         self.wt=nn.Linear(in_features=1024,out_features=hidden_dim)
#         self.ln=nn.LayerNorm(hidden_dim)
#         self.drop1=nn.Dropout(dropout)
#         self.drop2=nn.Dropout(dropout)
#         self.drop3=nn.Dropout(dropout)

#         self.proteinCompressor=ProteinCompressor(hidden_dim,num_head,rbp_feature_length,dropout)
    
#     def forward(self,x,embedding_mask,cellType):
#         x=self.drop1(F.relu(self.dense1(x)))
#         x=self.drop2(F.relu(self.dense2(x)))
#         x=self.drop3(self.wt(x))
#         x=self.ln(x)
#         cell_embedding=self.cell_embedding(cellType)
#         beta  = self.cell_beta_proj(cell_embedding)   # [B, 768]
#         gamma  = self.cell_gamma_proj(cell_embedding)   # [B, 768]
#         x = x * gamma.unsqueeze(1) + beta.unsqueeze(1)  # FiLM modulation
#         x_cls=x[:,0]
#         x=self.proteinCompressor(x,embedding_mask)
#         return x,x_cls

class Model_BC(nn.Module):

    def __init__(self,tokenizer,config,device):
        super().__init__()

         
        hidden_dim=config['d_model']
        cellTypeDim=config['cellTypeDim']
        num_head=config['num_head']
        dropout=config['dropout']
        self.momentum=config['momentum']
        rbp_feature_length=config['rbp_feature_length']
        cell_type_related=config['cell_type_related']
        vocab_size=config['vocab_size']
        stack_layers=config['stack_layers']
        kernel_sizes=config['kernel_sizes']
        dilations=config['dilations']

        self.mrna_fm, _ = fm.pretrained.mrna_fm_t12()
        self.rna_fm, _ = fm.pretrained.rna_fm_t12()

        
    

        bert_config = BertConfig.from_json_file(config['bert_config'])
        bert_config.vocab_size=len(tokenizer)
        self.text_encoder=BertForMaskedLM(config=bert_config)
        self.RNA_init=nn.Embedding(len(tokenizer),hidden_dim)
        self.RBP_enc=RBP_encoder(hidden_dim,cellTypeDim,num_head,rbp_feature_length,cell_type_related,dropout)
        self.RNA_encoder=RNA_encoder_fm(config['rna_fm_embedding_mode'],vocab_size,hidden_dim,stack_layers,num_head,kernel_sizes,dilations,dropout,device,)
        self.Linear_project = nn.Linear(hidden_dim,2)
        


        self.text_encoder_m=BertForMaskedLM(config=bert_config)
        self.RNA_init_m=nn.Embedding(len(tokenizer),hidden_dim)
        self.RBP_enc_m=RBP_encoder(hidden_dim,cellTypeDim,num_head,rbp_feature_length,cell_type_related,dropout)
        self.RNA_encoder_m=RNA_encoder_fm(config['rna_fm_embedding_mode'],vocab_size,hidden_dim,stack_layers,num_head,kernel_sizes,dilations,dropout,device,)
        self.Linear_project_m = nn.Linear(hidden_dim,2)
        self._init_momentum()


    @property
    def student(self):
        """返回所有学生模块（用于训练）"""
        return {
            'text_encoder': self.text_encoder,
            'RNA_init': self.RNA_init,
            'RBP_enc':self.RBP_enc,
            'RNA_encoder':self.RNA_encoder,
            'Linear_project':self.Linear_project,
            # 'Spatial_attention':self.ASTGNN
        }

    @property
    def teacher(self):
        """返回所有教师模块（用于生成伪标签）"""
        return {
            'text_encoder': self.text_encoder_m,
            'RNA_init': self.RNA_init_m,
            'RBP_enc':self.RBP_enc_m,
            'RNA_encoder':self.RNA_encoder_m,
            'Linear_project':self.Linear_project_m,
            # 'Spatial_attention':self.ASTGNN_m
        }

    @torch.no_grad()
    def _init_momentum(self):
        """复制参数并冻结教师"""
        for student_mod, teacher_mod in zip(self.student.values(), self.teacher.values()):
            teacher_mod.load_state_dict(student_mod.state_dict())
            for param in teacher_mod.parameters():
                param.requires_grad = False
            teacher_mod.eval()  # 初始化时设为 eval

    @torch.no_grad()
    def _momentum_update(self):
        for student_mod, teacher_mod in zip(self.student.values(), self.teacher.values()):
            for param, param_m in zip(student_mod.parameters(), teacher_mod.parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    
    def train(self, mode=True):
        super().train(mode)
        # 强制所有教师模块保持 eval
        for teacher_mod in self.teacher.values():
            teacher_mod.eval()
        return self

    def eval(self):
        super().eval()
        # 强制所有教师模块保持 eval（虽然它们已经是 eval，但显式强调）
        for teacher_mod in self.teacher.values():
            teacher_mod.eval()
        return self




    def forward(self,cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,label_mlm,seq_character,seq_character_mlm,rbp_embedding,label,alpha=0.0):
        
        self.mrna_fm.eval()  # disables dropout for deterministic results
        self.rna_fm.eval()  # disables dropout for deterministic results


        # 3. Extract embeddings (on CPU)
        with torch.no_grad():
            embedding_rna = self.rna_fm(batch_tokens_rna, repr_layers=[12])["representations"][12]
            embedding_mrna = self.mrna_fm(batch_tokens_mrna, repr_layers=[12])["representations"][12]
            # embedding_rna=None
            # embedding_mrna=None



        rbp_feat,rbp_feat_total=self.RBP_enc(rbp_embedding,embedding_mask,cellType)
        rbp_feat=F.normalize(rbp_feat,dim=-1)


        rna_emb,gate=self.RNA_encoder(embedding_rna,embedding_mrna,seq_character,rbp_feat_total)

        with torch.no_grad():
            self._momentum_update() #先动量编码器拷贝参数，momentum=0.995
            rbp_feat_m,rbp_feat_total_m=self.RBP_enc_m(rbp_embedding,embedding_mask,cellType)
            rbp_feat_m=F.normalize(rbp_feat_m,dim=-1)
            rna_emb_m,gate_m=self.RNA_encoder_m(embedding_rna,embedding_mrna,seq_character,rbp_feat_total_m) #获得RNA的全长的特征表示，这里交换最后两个维度是因为编码器当中存在CNN





        fusion=self.text_encoder.bert(encoder_embeds = rna_emb,
                                attention_mask = seq_attention_mask,
                                encoder_hidden_states = rbp_feat,
                                encoder_attention_mask = torch.ones((rbp_feat.shape[:2])),      
                                return_dict = True,
                                mode = 'fusion')

        profile=self.Linear_project(fusion.last_hidden_state[:,0,:])
        
        fusion_mlm=self.text_encoder(seq_character_mlm[label==1], 
                        attention_mask = seq_attention_mask[label==1],
                        encoder_hidden_states = rbp_feat[label==1],
                        encoder_attention_mask = torch.ones((rbp_feat.shape[:2])).to(rbp_feat.device)[label==1],  
                        labels = label_mlm[label==1],   
                        return_dict = True)
        mlm_output,loss_mlm_inner=fusion_mlm.logits,fusion_mlm.loss

        with  torch.no_grad():  
            fusion_m=self.text_encoder_m.bert(encoder_embeds = rna_emb_m, 
                                attention_mask = seq_attention_mask,
                                encoder_hidden_states = rbp_feat_m,
                                encoder_attention_mask = torch.ones((rbp_feat_m.shape[:2])),      
                                return_dict = True,
                                mode = 'fusion',)
            profile_m_raw=self.Linear_project_m(fusion_m.last_hidden_state[:,0,:])
        loss_profile=F.cross_entropy(profile,label)

        log_profile = F.log_softmax(profile, dim=-1)
        
        profile_m = F.softmax(profile_m_raw, dim=-1)
        profile_m = profile_m / profile_m.sum(dim=-1, keepdim=True)  # 归一化

        # 可选：加微小平滑避免目标中绝对零点带来的数值问题
        eps = 1e-8
        profile_m = profile_m.clamp(min=eps)
        profile_m = profile_m / profile_m.sum(dim=-1, keepdim=True)  # 重新归一化

        # 计算 KL 散度损失
        loss_KV_distill = F.kl_div(log_profile, profile_m, reduction='batchmean')*alpha

    
        return mlm_output,loss_profile+loss_KV_distill+loss_mlm_inner*0.1,profile,profile_m_raw
    
   
