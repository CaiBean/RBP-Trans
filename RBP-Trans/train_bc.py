import torch 
from dataset.dataset_bc import SequenceDataset4trainBC
from model.model_bc import Model_BC
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
from torchmetrics import MetricCollection, Accuracy, Precision, Recall,AUROC
import pandas as pd
from sklearn.model_selection import train_test_split # 导入 train_test_split 函数

# import pickle
# with open('/share/home/xuls/tongbu/Parallel-RBP/auc_list copy.pkl','rb')as f:
#     auc_list_1=pickle.load(f)
    



def validate_BC(model,val_dataloader,device,epoch):
    scaler = GradScaler()

    header = 'val Epoch: [{}]'.format(epoch)
    print_freq = 50 

    model.eval()

    metric_collection = MetricCollection({ 
            'acc': Accuracy(task="multiclass", num_classes=2) ,
            'prec': Precision(task="multiclass", num_classes=2,average='macro') ,
            'rec': Recall(task="multiclass", num_classes=2,average='macro') ,
            'auroc': AUROC(task="multiclass", num_classes=2)
        }) 
    
    metric_collection_m = MetricCollection({ 
            'acc': Accuracy(task="multiclass", num_classes=2) ,
            'prec': Precision(task="multiclass", num_classes=2,average='macro') ,
            'rec': Recall(task="multiclass", num_classes=2,average='macro') ,
            'auroc': AUROC(task="multiclass", num_classes=2)
        }) 

    metric_logger = MetricLogger(delimiter="  ")
  
    metric_logger.add_meter('acc', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('prec', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('rec', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('auroc', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('auroc_m', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_profile', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    profile_list,label_list,category_list=[],[],[]
    
    with torch.no_grad():

        for i,(cellType,sample_category,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,seq_character,seq_character_mlm,rbp_embedding,label_mlm,label) in enumerate(metric_logger.log_every(val_dataloader, print_freq, header)):
            cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,label,seq_attention_mask,seq_character,seq_character_mlm,label_mlm,rbp_embedding=cellType.to(device),batch_tokens_rna.to(device),batch_tokens_mrna.to(device),embedding_mask.to(device),label.to(device)\
                ,seq_attention_mask.to(device),seq_character.to(device),seq_character_mlm.to(device),label_mlm.to(device),rbp_embedding.to(device)
             
            alpha_tensor = torch.tensor([0.4], device=device)
            with torch.autocast(device_type="cuda"):   
                mlm_output,loss_profile,profile,profile_m=model(cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,label_mlm,seq_character,seq_character_mlm,rbp_embedding,label,alpha_tensor)
                                                       

                batch_metrics = metric_collection.forward(profile.detach().cpu(), label.cpu()) 
                batch_metrics_m = metric_collection_m.forward(profile_m.detach().cpu(), label.cpu()) 
  

    

        

            profile_list.extend(profile_m.tolist())
            label_list.extend(label.tolist())
            category_list.extend(sample_category)
            metric_logger.update(acc=batch_metrics['acc'])
            metric_logger.update(prec=batch_metrics['prec'])
            metric_logger.update(rec=batch_metrics['rec'])
            metric_logger.update(auroc=batch_metrics['auroc'])
            metric_logger.update(auroc_m=batch_metrics_m['auroc'])
            metric_logger.update(loss_profile=loss_profile)

            
    # gather the stats from all processes
    import pickle
    with open('auc_list_epoch_1_m.pkl', 'wb') as f:
        pickle.dump({'profile_list': profile_list, 'label_list': label_list,'category_list':category_list}, f)    
        print(f'保存成功')

    with open('auc_list_epoch_1_m.pkl', 'rb') as f:
        s=pickle.load( f)    
        profile_list,label_list,category_list=s['profile_list'],s['label_list'],s['category_list']
        print(f'保存成功')


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    



def train_BC(model,train_dataloader,optimizer,scheduler,warmup_steps,device,epoch):
    scaler = GradScaler()
    model.train()
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('alpha', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_profile', SmoothedValue(window_size=50, fmt='{value:.6f}'))

    # metric_logger.add_meter('KLDiv', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('prec', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('rec', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('auroc', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('auroc_m', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('acc_mlm', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50 




    for i,(cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,seq_character,seq_character_mlm,rbp_embedding,label_mlm,label) in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
        cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,label,seq_attention_mask,seq_character,seq_character_mlm,label_mlm,rbp_embedding=cellType.to(device),batch_tokens_rna.to(device),batch_tokens_mrna.to(device),embedding_mask.to(device),label.to(device)\
            ,seq_attention_mask.to(device),seq_character.to(device),seq_character_mlm.to(device),label_mlm.to(device),rbp_embedding.to(device)
            
        optimizer.zero_grad()

        
        metric_collection = MetricCollection({ 
            'acc': Accuracy(task="multiclass", num_classes=2) ,
            'prec': Precision(task="multiclass", num_classes=2,average='macro') ,
            'rec': Recall(task="multiclass", num_classes=2,average='macro') ,
            'auroc': AUROC(task="multiclass", num_classes=2)
        }) 
        metric_collection_m = MetricCollection({ 
            'acc': Accuracy(task="multiclass", num_classes=2) ,
            'prec': Precision(task="multiclass", num_classes=2,average='macro') ,
            'rec': Recall(task="multiclass", num_classes=2,average='macro') ,
            'auroc': AUROC(task="multiclass", num_classes=2)
        }) 




        alpha = config['alpha']*min(1,i/len(train_dataloader))  if epoch==1 else 0.4
        alpha_tensor = torch.tensor([alpha], device=device) if epoch!=0 else torch.tensor([0.0], device=device)
        with torch.autocast(device_type="cuda"):   
            mlm_output,loss_profile,profile,profile_m=model(cellType,batch_tokens_rna,batch_tokens_mrna,embedding_mask,seq_attention_mask,label_mlm,seq_character,seq_character_mlm,rbp_embedding,label,alpha_tensor)
                    

        


        label_mlm=label_mlm[label==1]

        # acc_mlm=sum((mlm_output[label==1].detach()[label_mlm!=-100]).argmax(-1)==label_mlm[label_mlm!=-100]).item()/len(label_mlm[label_mlm!=-100]) if sum(label)>0 else 0
        acc_mlm=sum((mlm_output.detach()[label_mlm!=-100]).argmax(-1)==label_mlm[label_mlm!=-100]).item()/len(label_mlm[label_mlm!=-100]) 

        batch_metrics = metric_collection.forward(profile.detach().cpu(), label.cpu()) 
        batch_metrics_m = metric_collection_m.forward(profile_m.detach().cpu(), label.cpu())

        loss = loss_profile

    
        scaler.scale(loss.half()).backward()
        scaler.step(optimizer)
        scaler.update()   
        
        metric_logger.update(alpha=alpha_tensor.item())
        metric_logger.update(acc_mlm=acc_mlm)
        metric_logger.update(loss_profile=loss_profile)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])     
        # metric_logger.update(KLDiv=loss_m)
        
        metric_logger.update(acc=batch_metrics['acc'])
        metric_logger.update(prec=batch_metrics['prec'])
        metric_logger.update(rec=batch_metrics['rec'])
        metric_logger.update(auroc=batch_metrics['auroc'])


        metric_logger.update(auroc_m=batch_metrics_m['auroc'])

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
    

    df=pd.read_parquet(args.parquet_path)
    assert args.parquet_path in ['data/Reformer_bc.parquet','data/RBP42Dataset.parquet']
    if args.parquet_path=='data/Reformer_bc.parquet':
        train_df,test_df =df[df['mode']=='trn'],df[df['mode']=='val']
        mode='Reformer'
    else:
        train_df,test_df = train_test_split(df, test_size=0.2, random_state=42)
        mode='RBP42'

    print("Creating model")
    model = Model_BC(tokenizer,config,mode)
    model = model.to(device)

    arg_opt = AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)  
    arg_sche = AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    start_epoch = 0
    if args.checkpoint:
        if args.resume:
            checkpoint = torch.load(args.checkpoint, map_location='cpu',weights_only=True)
            model.load_state_dict(checkpoint['model'],strict=False)
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
        
        model.load_state_dict(checkpoint,strict=True)
        print('load checkpoint from %s' % args.checkpoint)

    # # 编译模型（放在最后，所有准备就绪）
    model = torch.compile(model)


    '''
        训练模型
    '''
    print("Creating dataset")
    val_dataset= SequenceDataset4trainBC(test_df,config['rbp_embedding'],tokenizer,max_length=512,mode='val')
    train_dataset=SequenceDataset4trainBC(train_df,config['rbp_embedding'],tokenizer,max_length=512,mode='trn')

   
    


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
        train_dataloader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,num_workers=16,persistent_workers=True,drop_last=True,pin_memory=True)    # 数据加载器
        val_dataloader=DataLoader(val_dataset,batch_size=config['batch_size'],shuffle=False,num_workers=16,persistent_workers=True,drop_last=False,pin_memory=True) 

    


    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config['epochs']):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
        # val_stats=validate_BC(model,val_dataloader,device,epoch)
        train_stats=train_BC(model,train_dataloader,optimizer,lr_scheduler,warmup_steps,device,epoch)
        
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
        
        val_stats=validate_BC(model,val_dataloader,device,epoch)
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
    parser.add_argument('--config', default='configs/bc_config.yaml')
    parser.add_argument('--output_dir', default='checkpoint/bc')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', default="") #checkpoint/bc_42_dataset/checkpoint_19.pth
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--parquet_path', help='input parquet file',default='data/Reformer_bc.parquet')#/share/home/xuls/tongbu/mnist-clip-main/data/Reformer_bc.parquet




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