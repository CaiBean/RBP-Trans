from layers.layers import *
from layers.xbert import *

class Model_Rank(nn.Module):

    def __init__(self,tokenizer,config):
        super().__init__()


        hidden_dim=config['hidden_dim']
        cellTypeDim=config['cellTypeDim']
        kernel_sizes=config['kernel_sizes_42RBP']
        dilations=config['dilations_42RBP']
        num_head=config['num_head']
        dropout=config['dropout']
        self.momentum=config['momentum']


        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder=BertForMaskedLM(config=bert_config)
        self.rnaSeq_encoder=BertForMaskedLM(config=bert_config)
        self.RNA_init=nn.Embedding(len(tokenizer),hidden_dim)
        self.rbp_enc=RBP_encoder(hidden_dim,cellTypeDim)
        self.RNA_encoder=RNA_encoder(4,num_head,hidden_dim,kernel_sizes,dilations,dropout)

        self.Linear_project_p_value = nn.Linear(hidden_dim,1)
        self.Linear_project_signal_value = nn.Linear(hidden_dim,1)




        self.text_encoder_m=BertForMaskedLM(config=bert_config)
        self.rnaSeq_encoder_m=BertForMaskedLM(config=bert_config)
        self.RNA_init_m=nn.Embedding(len(tokenizer),hidden_dim)
        self.rbp_enc_m=RBP_encoder(hidden_dim,cellTypeDim)
        self.RNA_encoder_m=RNA_encoder(4,num_head,hidden_dim,kernel_sizes,dilations,dropout)

        self.Linear_project_p_value_m = nn.Linear(hidden_dim,1)
        self.Linear_project_signal_value_m = nn.Linear(hidden_dim,1)
        self._init_momentum()


                


    @property
    def student(self):
        """返回所有学生模块（用于训练）"""
        return {
            'xbert_encoder': self.text_encoder,
            'RNA_init': self.RNA_init,
            'RBP_enc':self.rbp_enc,
            'RNA_encoder':self.RNA_encoder,
            'Linear_project_p_value':self.Linear_project_p_value,
            'Linear_project_signal_value':self.Linear_project_signal_value

        }

    @property
    def teacher(self):
        """返回所有教师模块（用于生成伪标签）"""
        return {
            'xbert_encoder': self.text_encoder_m,
            'RNA_init': self.RNA_init_m,
            'RBP_enc':self.rbp_enc_m,
            'RNA_encoder':self.RNA_encoder_m,
            'Linear_project_p_value':self.Linear_project_p_value_m,
            'Linear_project_signal_value':self.Linear_project_signal_value_m
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
    
    def forward(self,cellType,seq_attention_mask,seq_character,seq_character_mlm,rbp_embedding,label_mlm,signal_value_norm,p_value_norm,alpha=0.0):



        rbp_feat=F.normalize(self.rbp_enc(rbp_embedding,cellType),dim=-1)


        rna_emb_mlm=self.RNA_init(seq_character_mlm)

        rna_emb_mlm=self.RNA_encoder(rna_emb_mlm.permute(0,2,1))
        
        
        rna_emb=self.RNA_init(seq_character)

        rna_emb=self.RNA_encoder(rna_emb.permute(0,2,1))

        # rna_feat=F.normalize(rna_emb[:,0,:],dim=-1) #融合前的embedding,用于ITC
        # itc_1,itc_2=rna_feat@rbp_feat.T/self.temprature,rna_feat@rbp_feat.T/self.temprature


        with torch.no_grad():
            self._momentum_update() #先动量编码器拷贝参数，momentum=0.995
            rbp_feat_m=F.normalize(self.rbp_enc_m(rbp_embedding,cellType),dim=-1)

 

            rna_emb_m=self.RNA_init_m(seq_character)#动量编码器内部对tokenizer处理的rna序列进行embedding，尺寸：(batch_size，length,hidden_dim)

            rna_emb_m=self.RNA_encoder_m(rna_emb_m.permute(0,2,1)) #获得RNA的全长的特征表示，这里交换最后两个维度是因为编码器当中存在CNN
        #   rna_feat_m=rna_emb_m[:,0,:] #提取cls位置的embedding作为整个rna的特征向量，
        
        #     sim_s2p_m =  rna_feat_m @ rbp_feat_m.T/ self.temprature #s2p表示sequence2protein，也就是rna对rbp，温度系数0.07
        #     sim_p2s_m=  rbp_feat_m @ rna_feat_m.T/self.temprature 
   

        #     sim_s2p_targets = alpha * F.softmax(sim_s2p_m, dim=1) + (1 - alpha) * mask  #此处的mask为函数外部传入真实结合矩阵，尺寸为(RNA个数，RBP个数)，1表示对应RBP与RNA结合，0表示不结合
        #     sim_p2s_targets = alpha * F.softmax(sim_p2s_m, dim=1) + (1 - alpha) * (mask.T)     #动量编码器内部生成标签，以减少真实标签的误差，

        # sim_s2p =  rna_feat @ rbp_feat.T/ self.temprature 
        # sim_p2s=  rbp_feat @ rna_feat.T/self.temprature 
        
        # log_probs_s2p = sim_s2p.log_softmax(dim=1)
        # log_probs_p2s = sim_p2s.log_softmax(dim=1)
        
                             
        # loss_s2p = - (log_probs_s2p * sim_s2p_targets).sum(dim=1) / sim_s2p_targets.sum(dim=1).clamp(min=1)
        # loss_p2s = - (log_probs_p2s * sim_p2s_targets).sum(dim=1) / sim_p2s_targets.sum(dim=1).clamp(min=1)


        # loss_itc = (loss_p2s.mean()+loss_s2p.mean())/2



        fusion=self.text_encoder.bert(encoder_embeds = rna_emb, 
                                attention_mask = seq_attention_mask,
                                encoder_hidden_states = rbp_feat.unsqueeze(1),
                                encoder_attention_mask = torch.ones((rbp_feat.unsqueeze(1).shape)),      
                                return_dict = True,
                                mode = 'fusion',)
        profile_p_value=F.sigmoid(self.Linear_project_p_value(fusion.last_hidden_state[:,0,:]).squeeze(-1))
        profile_signal_value=F.sigmoid(self.Linear_project_signal_value(fusion.last_hidden_state[:,0,:]).squeeze(-1))
        
        fusion_mlm=self.text_encoder(seq_character_mlm, 
                        attention_mask = seq_attention_mask,
                        encoder_hidden_states = rbp_feat.unsqueeze(1),
                        encoder_attention_mask = torch.ones((rbp_feat.unsqueeze(1).shape)),
                        labels = label_mlm,   
                        return_dict = True)
        mlm_output,loss_mlm_inner=fusion_mlm.logits,fusion_mlm.loss
        with  torch.no_grad():  
            fusion_m=self.text_encoder_m.bert(encoder_embeds = rna_emb_m, 
                                attention_mask = seq_attention_mask,
                                encoder_hidden_states = rbp_feat_m.unsqueeze(1),
                                encoder_attention_mask = torch.ones((rbp_feat_m.unsqueeze(1).shape)),      
                                return_dict = True,
                                mode = 'fusion',)
            
            profile_p_value_m=F.sigmoid(self.Linear_project_p_value_m(fusion_m.last_hidden_state[:,0,:]).squeeze(-1))
            profile_signal_value_m=F.sigmoid(self.Linear_project_signal_value_m(fusion_m.last_hidden_state[:,0,:]).squeeze(-1))
            p_value_norm_label=p_value_norm*(1-alpha)+alpha*profile_p_value_m
            signal_value_norm_label=signal_value_norm*(1-alpha)+alpha*profile_signal_value_m

        loss_p_value=F.l1_loss(profile_p_value,p_value_norm_label)
        loss_signal_value=F.l1_loss(profile_signal_value,signal_value_norm_label)


        return mlm_output,loss_signal_value,loss_p_value,loss_mlm_inner,profile_p_value_m,profile_signal_value_m,profile_p_value,profile_signal_value
    