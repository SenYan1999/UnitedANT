import torch
import pandas as pd
import math

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
    print('loading train data...')
    train_dataset = torch.load(args.train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    print('loading dev data...')
    dev_dataset = torch.load(args.dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare model
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
    model = MultimodalityModel(args.longformer_name, args.audio_dim, args.hidden_size, args.num_layers,\
         args.n_head, args.table_in_dim, args.table_out_dim, args.drop_out, device)
    model = model.to(device)

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # apex distributed training
    # to be continue

    trainer = Trainer(train_dataloader, dev_dataloader, model, optimizer, device, args.tau, args.fp16)
    trainer.train(args.num_epoch, args.save_path)

if __name__ == "__main__":
    if args.do_prepare:
        do_prepare()
    elif args.do_train:
        do_train()
    else:
        print('Nothing have done.')
