import torch
import torch.nn as nn
from .LSTM import BiLSTM
from .SelfAttention import MultiHeadAttention

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.1, bidirectional=True, atten_head=4):
        super(BiLSTMEncoder, self).__init__()
        self.lstm_encoder = BiLSTM(input_size, hidden_size // 2, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.atten = MultiHeadAttention(atten_head, hidden_size, hidden_size, hidden_size, dropout=dropout)

    def forward(self, x):
        lstm_out = self.lstm_encoder(x)
        atten_out = self.atten(lstm_out, lstm_out, lstm_out)
        pooler_out = nn.functional.max_pool1d(atten_out.transpose(-2, -1), kernel_size=atten_out.shape[-2]).squeeze(dim=-1)
        return pooler_out
