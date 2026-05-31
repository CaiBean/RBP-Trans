from layers.layers import *
from layers.xbert import *
import fm


    

class Model_Base(nn.Module):

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
        self.RBP_enc=RBP_encoder(hidden_dim,cellTypeDim,num_head,rbp_feature_length,cell_type_related,dropout)
        self.RNA_encoder=RNA_encoder_fm(config['rna_fm_embedding_mode'],vocab_size,hidden_dim,stack_layers,num_head,kernel_sizes,dilations,dropout,device,)
        self.Linear_project_m = nn.Linear(hidden_dim,1)
        self.Linear_project = nn.Linear(hidden_dim,1)


        self.text_encoder_m=BertForMaskedLM(config=bert_config)
        self.RBP_enc_m=RBP_encoder(hidden_dim,cellTypeDim,num_head,rbp_feature_length,cell_type_related,dropout)
        self.RNA_encoder_m=RNA_encoder_fm(config['rna_fm_embedding_mode'],vocab_size,hidden_dim,stack_layers,num_head,kernel_sizes,dilations,dropout,device,)
        self.Linear_project_m = nn.Linear(hidden_dim,1)

        self._init_momentum()

    @property
    def student(self):
        """返回所有学生模块（用于训练）"""
        return {
            'text_encoder': self.text_encoder,
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
    


    def forward(self,cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,label_mlm,seq_attention_mask,seq_character,seq_character_mlm,rbp_embedding,rbp_combine_profile,alpha):
        self.mrna_fm.eval()  # disables dropout for deterministic results
        self.rna_fm.eval()  # disables dropout for deterministic results


        # 3. Extract embeddings (on CPU)
        
        with torch.no_grad():
            embedding_rna = self.rna_fm(batch_tokens_rna, repr_layers=[12])["representations"][12]
            embedding_mrna = self.mrna_fm(batch_tokens_mrna, repr_layers=[12])["representations"][12]
            # embedding_mrna=None
            # embedding_rna=None



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

        profile=self.Linear_project(fusion.last_hidden_state).squeeze(-1)



        fusion_mlm=self.text_encoder(seq_character_mlm, 
                        attention_mask = seq_attention_mask,
                        encoder_hidden_states = rbp_feat,
                        encoder_attention_mask = torch.ones((rbp_feat.shape[:2])),  
                        labels = label_mlm,   
                        return_dict = True)
        mlm_output,loss_mlm_inner=fusion_mlm.logits,fusion_mlm.loss
        
        with  torch.no_grad():  
            fusion_m=self.text_encoder_m.bert(encoder_embeds = rna_emb_m, 
                                attention_mask = seq_attention_mask,
                                encoder_hidden_states = rbp_feat_m,
                                encoder_attention_mask = torch.ones((rbp_feat_m.shape[:2])),      
                                return_dict = True,
                                mode = 'fusion',)
            profile_m_raw=self.Linear_project_m(fusion_m.last_hidden_state).squeeze(-1)


        loss_l1=0.05*F.l1_loss(profile,rbp_combine_profile)
        log_profile = F.log_softmax(profile, dim=-1)
        

        # 标签归一化：target_counts 是原始 eCLIP 信号强度 (B, 512)
        profile_m = profile_m_raw / profile_m_raw.sum(dim=-1, keepdim=True)  # 归一化

        # 可选：加微小平滑避免目标中绝对零点带来的数值问题
        eps = 1e-8
        profile_m = profile_m.clamp(min=eps)
        profile_m = profile_m / profile_m.sum(dim=-1, keepdim=True)  # 重新归一化

        # 计算 KL 散度损失
        loss_KV_distill = F.kl_div(log_profile, profile_m, reduction='batchmean')*alpha




        

        # 标签归一化：target_counts 是原始 eCLIP 信号强度 (B, 512)
        rbp_combine_profile = rbp_combine_profile / rbp_combine_profile.sum(dim=-1, keepdim=True)  # 归一化

        # 可选：加微小平滑避免目标中绝对零点带来的数值问题
        eps = 1e-8
        rbp_combine_profile = rbp_combine_profile.clamp(min=eps)
        rbp_combine_profile = rbp_combine_profile / rbp_combine_profile.sum(dim=-1, keepdim=True)  # 重新归一化

        # 计算 KL 散度损失
        loss_KV = F.kl_div(log_profile, rbp_combine_profile, reduction='batchmean')



        return mlm_output,loss_l1+loss_KV+loss_KV_distill,profile,profile_m_raw

