import torch
import torch.nn as nn
import torch.multiprocessing as mp
import pandas as pd
import math
import os

from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from model import MultimodalityModel
from torch.utils.data import DataLoader
from utils import MultimodalityDataset, Trainer
from args import args

def do_prepare():
    # prepare company to idx
    company2idx = torch.load(args.company2idx)

    # prepare table data 
    table = pd.read_csv(args.table_file)

    # train dataset
    train_dataset = MultimodalityDataset(company2idx, args.audio_dir, table, args.text_dir, args.text_max_len,\
         0, -math.inf, args.split_date, args.longformer_name)
    torch.save(train_dataset, args.train_dataset)

    # dev dataset
    dev_dataset = MultimodalityDataset(company2idx, args.audio_dir, table, args.text_dir, args.text_max_len, \
         0, args.split_date, math.inf, args.longformer_name)
    torch.save(dev_dataset, args.dev_dataset)

def do_train():
    # prepare dataset
    print('[Prepare] Loading train data...')
    train_dataset = torch.load(args.train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    print('[Prepare] Loading dev data...')
    dev_dataset = torch.load(args.dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare model
    print('[Prepare] Construct model')
    audio_options = {
        'ckpt_file'     : './pretrained_model/states-1500000.ckpt',
        'load_pretrain' : 'True',
        'no_grad'       : 'True',
        'dropout'       : 'default',
        'spec_aug'      : 'False',
        'spec_aug_prev' : 'True',
        'weighted_sum'  : 'False',
        'select_layer'  : -1,
    }
    model = MultimodalityModel(args.longformer_name, audio_options, args.table_in_dim, args.table_out_dim, device)
    model = model.to(device)
    # model = model.to(device)
    # model = DistributedMultimodalityModel(args.longformer_name, audio_options, args.table_in_dim, args.table_out_dim, device)

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.parallel:
        model = nn.DataParallel(model)
        model = model.cuda()

    # apex distributed training
    # to be continue

    trainer = Trainer(train_dataloader, dev_dataloader, model, optimizer, args.fp16)
    trainer.train(args.num_epoch, args.save_path)

def do_train_distributed(gpu, _):
    # prepare distributed 
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    rank = args.nr * args.gpus + gpu	                          
    world_size = args.gpus * args.nodes
    torch.distributed.init_process_group(                                   
        backend='nccl',                                         
   	    init_method='env://',                                   
        world_size=world_size,                              
        rank=rank                                               
    )

    # prepare dataset
    print('[Prepare] Loading train data...')
    train_dataset = torch.load(args.train_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    print('[Prepare] Loading dev data...')
    dev_dataset = torch.load(args.dev_dataset)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, sampler=dev_sampler)

    # prepare model
    print('[Prepare] Construct model')
    audio_options = {
        'ckpt_file'     : './pretrained_model/states-1500000.ckpt',
        'load_pretrain' : 'True',
        'no_grad'       : 'True',
        'dropout'       : 'default',
        'spec_aug'      : 'False',
        'spec_aug_prev' : 'True',
        'weighted_sum'  : 'False',
        'select_layer'  : -1,
    }
    model = MultimodalityModel(args.longformer_name, audio_options, args.table_in_dim, args.table_out_dim, None)
    model.cuda(gpu)

    # prepare optimizer
    text_param = list(filter(lambda kv: kv[0].startswith('text_encoder'), model.named_parameters()))
    audio_param = list(filter(lambda kv: kv[0].startswith('audio_encoder'), model.named_parameters()))
    rest_param = list(filter(lambda kv: (not kv[0].startswith('text_encoder') and (not kv[0].startswith('audio_encoder'))), model.named_parameters()))
    optimizer = torch.optim.Adam([
        {'params': text_param, 'lr': 2e-5},
        {'params': audio_param, 'lr': 2e-5},
        {'params': rest_param, 'lr': 2e-5}
    ])

    # distribution training
    model, optimizer = amp.initialize(model, optimizer, 
                                    opt_level='O2')
    model = DDP(model)

    trainer = Trainer(train_dataloader, dev_dataloader, model, optimizer, args.fp16)
    trainer.train(args.num_epoch, args.save_path, dist=True)

if __name__ == "__main__":
    if args.do_prepare:
        do_prepare()
    elif args.do_train:
        if not args.distributed:
            do_train()
        else:
            args.world_size = args.gpus * args.nodes                
            os.environ['MASTER_ADDR'] = 'gpu02'              
            os.environ['MASTER_PORT'] = '4318'                      
            mp.spawn(do_train_distributed, nprocs=args.gpus, args=(args,))
    else:
        print('Nothing have done.')
