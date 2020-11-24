from argparse import ArgumentParser

parser = ArgumentParser()

# train mode
parser.add_argument('--do_prepare', action='store_true')
parser.add_argument('--do_train', action='store_true')

# data_prepare
parser.add_argument('--company2idx', type=str, default='/data1/yansen/wits/data/company2idx.pt')
parser.add_argument('--audio_dir', type=str, default='/data1/yansen/wits/data/audios')
parser.add_argument('--text_dir', type=str, default='/data1/yansen/wits/data/texts')
parser.add_argument('--table_file', type=str, default='/data1/yansen/wits/data/volatility20200725.csv')
parser.add_argument('--split_date', type=int, default=20171101)
parser.add_argument('--text_max_len', type=int, default=4096)
parser.add_argument('--train_dataset', type=str, default='/data1/yansen/wits/data/train.pt')
parser.add_argument('--dev_dataset', type=str, default='/data1/yansen/wits/data/dev.pt')

# config of model
parser.add_argument('--longformer_name', type=str, default='allenai/longformer-base-4096')
parser.add_argument('--audio_dim', type=int, default=6373)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--table_in_dim', type=int, default=1)
parser.add_argument('--table_out_dim', type=int, default=128)

# train setting
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--tau', type=int, default=3)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--num_epoch', type=int, default=12)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--fp16', action='store_true')

# save and log
parser.add_argument('--save_path', type=str, default='output')

args = parser.parse_args()