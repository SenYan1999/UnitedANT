import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.1, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.enc = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, \
            batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, x):
        # x: shape(batch_size, seq_len, seq_dim)
        x_len = torch.sum(torch.sum(x != 0, dim=-1).gt(0), dim=-1)
        origin_len = x.shape[1]
        lengths, sorted_idx = x_len.sort(0, descending=True)
        x = x[sorted_idx]
        inp = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.enc(inp)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=origin_len)
        _, unsorted_idx = sorted_idx.sort(0)
        out = out[unsorted_idx]
        return out