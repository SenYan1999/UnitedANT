import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    # def __init__(self, input_size, hidden_size, bidirectional, num_layers, batch_first):
    def __init__(self, **args):
        super(LSTM, self).__init__()
        # self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.lstm = nn.LSTM(**args)
    
    def forward(self, x, x_len=None):
        # sort the sequence in descending order
        if not x_len:
            x_len = torch.sum((torch.sum(x, dim=-1) != 0).int(), dim=-1)
        seq_len, perm_idx = x_len.sort(0, descending=True)
        sorted_x = x[perm_idx]

        # pack the padded sequences
        pack_seq = pack_padded_sequence(sorted_x, seq_len, batch_first=True)

        # apply lstm
        lstm_out, _ = self.lstm(pack_seq)

        # pad the packed sequences
        pad_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=x.shape[1])

        # sort the pad out to its original order
        _, unsorted_idx = perm_idx.sort(0)
        out = pad_out[unsorted_idx]
        
        # out, _ = self.lstm(x)

        return out

