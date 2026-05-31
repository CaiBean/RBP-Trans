import torch 
from dataset.dataset_base import SequenceDataset4train
from model.model_base import Model_Base
from torch.utils.data import DataLoader
import time
import datetime
import json
import numpy as np
import random
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
import pandas as pd
torch.set_float32_matmul_precision('high')




def validate_base(model,val_dataloader,device,epoch):
    scaler = GradScaler()

    header = 'val Epoch: [{}]'.format(epoch)
    print_freq = 1

    model.eval()



    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('iou', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    # metric_logger.add_meter('acc_mlm', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Spearman_Corr', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Pearson_Corr_individual', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Pearson_Corr_m_individual', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Spearman_Corr_individual', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Spearman_Corr_m_individual', SmoothedValue(window_size=50, fmt='{value:.4f}'))

    profile_list,profile_m_list,spearman_list,pearson_list,category_list=[],[],[],[],[]
    spearman_m_list,pearson_m_list=[],[]
    label_list=[]
    pos_list=[]

    with torch.no_grad():

        for i,(cellType,category,chr,peak_start,peak_end,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,seq_character,seq_character_mlm,rbp_embedding,label_mlm,label) in  enumerate(metric_logger.log_every(val_dataloader, print_freq, header)):
            cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,label,seq_attention_mask,seq_character,seq_character_mlm,label_mlm,rbp_embedding=cellType.to(device),batch_tokens_rna.to(device),batch_tokens_mrna.to(device),embedding_mask.to(device),label.to(device)\
                ,seq_attention_mask.to(device),seq_character.to(device),seq_character_mlm.to(device),label_mlm.to(device),rbp_embedding.to(device)

            alpha = config['alpha']*min(1,i/len(val_dataloader))  if epoch==0 else 0.4
            alpha_tensor = torch.tensor([0.0], device=device)
            with torch.autocast(device_type="cuda"):
                mlm_output,loss_profile,profile,profile_m=model(cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,label_mlm,seq_character,seq_character_mlm,rbp_embedding,label,alpha_tensor)
                
            input=(cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,label_mlm,seq_character,seq_character_mlm,rbp_embedding,label,alpha_tensor)
            profile_list.append(profile.detach().tolist()[0])
            profile_m_list.append(profile_m.detach().tolist()[0])
            label_list.append(label.detach().cpu().tolist()[0])


            # mlm_acc=sum((mlm_output.detach()[label_mlm!=-100]).argmax(-1)==label_mlm[label_mlm!=-100]).item()/len(label_mlm[label_mlm!=-100])
            correlations = compute_correlations_total(profile.detach(), label)
            correlations_individual = compute_correlations_individual(profile.detach().to(torch.float32), label.to(torch.float32))
            correlations_m_individual = compute_correlations_individual(profile_m.detach().to(torch.float32),label.to(torch.float32))

            spearman_list.append(correlations_individual['Spearman_Correlation'].detach().tolist())
            pearson_list.append(correlations_individual['Pearson_Correlation'].detach().tolist())
            spearman_m_list.append(correlations_m_individual['Spearman_Correlation'].detach().tolist())
            pearson_m_list.append(correlations_m_individual['Pearson_Correlation'].detach().tolist())
            category_list.append(category[0])
            pos_list.append(chr[0]+':'+str(peak_start[0])+'-'+str(peak_end[0]))

            
            metric_logger.update(iou=getIOU(profile.detach(),label).mean().item())
            # metric_logger.update(acc_mlm=mlm_acc)
            metric_logger.update(Spearman_Corr=round(correlations['Spearman_Correlation'],4))
            metric_logger.update(Spearman_Corr_individual=round(correlations_individual['Spearman_Correlation'].detach().cpu().item(),4))
            metric_logger.update(Spearman_Corr_m_individual=round(correlations_m_individual['Spearman_Correlation'].detach().cpu().item(),4))  
            metric_logger.update(Pearson_Corr_individual=round(correlations_individual['Pearson_Correlation'].detach().cpu().item(),4))    
            metric_logger.update(Pearson_Corr_m_individual=round(correlations_m_individual['Pearson_Correlation'].detach().cpu().item(),4))    
            import pickle
            with open('aa.pkl','wb')as f:
                pickle.dump([profile_list,profile_m_list,spearman_list,pearson_list,spearman_m_list,pearson_m_list,category_list,label_list,pos_list],f)


            import pickle
            with open('aa.pkl','rb')as f:
                profile_list,profile_m_list,spearman_list,pearson_list,spearman_m_list,pearson_m_list,category_list,label_list,pos_list=pickle.load(f)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    



def train_base(model,train_dataloader,optimizer,scheduler,warmup_steps,device,epoch):
    scaler = GradScaler()
    model.train()
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('iou', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Spearman_Corr_individual', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Spearman_Corr_individual_m', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Spearman_Corr', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Pearson_Corr', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('acc_mlm', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('alpha', SmoothedValue(window_size=50, fmt='{value:.4f}'))


    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50 




    for i,(cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,seq_character,seq_character_mlm,rbp_embedding,label_mlm,label) in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
        cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,label,seq_attention_mask,seq_character,seq_character_mlm,label_mlm,rbp_embedding=cellType.to(device),batch_tokens_rna.to(device),batch_tokens_mrna.to(device),embedding_mask.to(device),label.to(device)\
            ,seq_attention_mask.to(device),seq_character.to(device),seq_character_mlm.to(device),label_mlm.to(device),rbp_embedding.to(device)
        optimizer.zero_grad()

        



        alpha = config['alpha']*min(1,i/len(train_dataloader))  if epoch==0 else 0.4
        alpha_tensor = torch.tensor([alpha], device=device) 
        with torch.autocast(device_type="cuda"):        
            mlm_output,loss_profile,profile,profile_m=model(cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,label_mlm,seq_character,seq_character_mlm,rbp_embedding,label,alpha_tensor)

        acc_mlm=sum((mlm_output.detach()[label_mlm!=-100]).argmax(-1)==label_mlm[label_mlm!=-100]).item()/len(label_mlm[label_mlm!=-100]) 


        correlations = compute_correlations_total(profile.to(label.dtype).detach(), label)
        # correlations_m = compute_correlations_total(profile_m.to(rbp_combine_profile.dtype).detach(),rbp_combine_profile)
        correlations_individual = compute_correlations_individual(profile.to(label.dtype).detach(), label)
        correlations_m_individual = compute_correlations_individual(profile_m.to(label.dtype).detach(),label)

        loss = loss_profile

    
        scaler.scale(loss.half()).backward()
        scaler.step(optimizer)
        scaler.update()   

        
        metric_logger.update(acc_mlm=acc_mlm)
        metric_logger.update(alpha=alpha_tensor.item())
        metric_logger.update(iou=getIOU(profile.detach(),label).mean().item())
        metric_logger.update(Spearman_Corr=round(correlations['Spearman_Correlation'],4))

        metric_logger.update(Spearman_Corr_individual=round(correlations_individual['Spearman_Correlation'].detach().cpu().item(),4))
        metric_logger.update(Spearman_Corr_individual_m=round(correlations_m_individual['Spearman_Correlation'].detach().cpu().item(),4))
        metric_logger.update(Pearson_Corr=round(correlations['Pearson_Correlation'],4))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         



        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    
    


def main(args, config):


    device=torch.device(f'{args.cuda}' if torch.cuda.is_available() else 'cpu' )  # 设备
    print(f'device:{device}')
    tokenizer = T.BertTokenizer.from_pretrained(config['tokenizer_path'])
    warmup_steps = config['schedular']['warmup_epochs']    
    start_epoch=0

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    



    print("Creating model")
    model = Model_Base(tokenizer,config,device)
    model = model.to(device)

    arg_opt = AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)  
    arg_sche = AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    start_epoch = 0
    if args.checkpoint:
        if args.resume:
            checkpoint = torch.load(args.checkpoint, map_location='cpu',weights_only=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        # 加载模型权重到原始模型
        checkpoint = torch.load(args.checkpoint, map_location='cpu',weights_only=True)['model']
        keys_list = list(checkpoint.keys())
        for key in keys_list:
            if 'orig_mod.' in key:
                deal_key = key.replace('_orig_mod.', '')
                checkpoint[deal_key] = checkpoint[key]
                del checkpoint[key]
        
        model.load_state_dict(checkpoint,strict=False)
        print('load checkpoint from %s' % args.checkpoint)

    # 编译模型（放在最后，所有准备就绪）
    model = torch.compile(model)


    '''
        训练模型
    '''
    print("Creating dataset")
    
    data_total=pd.read_parquet("/share/home/xuls/dataset/eclipDataFromEncodeWithNewCome/totol_rep1.parquet")
    mode_list=[np.random.choice(['trn', 'val','test'], p=[0.9, 0.09,0.01]) for i in range(len(data_total))]
    data_total['mode']=mode_list
    val_dataset= SequenceDataset4train(data_total,args.eclip_path,config['rbp_embedding'],config['rbp_list'],tokenizer,max_length=512,mode='val')
    train_dataset=SequenceDataset4train(data_total,args.eclip_path,config['rbp_embedding'],config['rbp_list'],tokenizer,max_length=512,mode='trn')

   
    


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
        val_dataloader=DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=1,persistent_workers=True,drop_last=False,pin_memory=True) 

    


    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config['epochs']):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
        val_stats=validate_base(model,val_dataloader,device,epoch)
        # train_stats=train_base(model,train_dataloader,optimizer,lr_scheduler,warmup_steps,device,epoch)
        
        if is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),  # 保存原始模型
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config':config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            print('save checkpoint_%02d.pth successfully'%epoch )
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        val_stats=validate_base(model,val_dataloader,device,epoch)
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
    parser.add_argument('-c','--cuda',default='cuda')
    parser.add_argument('--config', default='configs/base_config.yaml')
    parser.add_argument('--output_dir', default='checkpoint/base_DRAFT')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', default="/share/home/xuls/tongbu/Parallel-RBP/checkpoint/base/checkpoint_00.pth") #checkpoint/bc_42_dataset/checkpoint_19.pth
    parser.add_argument('--resume', default=True, type=bool)
    parser.add_argument('--eclip_path', help='input parquet file',default='/share/home/xuls/tongbu/Parallel-RBP/data/encode_eclip.h5')#/share/home/xuls/tongbu/mnist-clip-main/data/Reformer_bc.parquet




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