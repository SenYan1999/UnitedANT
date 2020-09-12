import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig
from .Attention import Attention
from .RNN import LSTM

class LongformerEncoder(nn.Module):
    def __init__(self, model_name):
        super(LongformerEncoder, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.gradient_checkpointing = True
        self.longformer = AutoModel.from_pretrained(model_name, config=self.config)
        self.rnn = LSTM(input_size=768, hidden_size=768 // 2, bidirectional=True, num_layers=1, batch_first=True)
        self.select_idx = torch.tensor([i * 64 for i in range(0, 64)]).cuda()
        # self.conv1d = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=128, stride=32) # 125
        # self.attention = Attention(dimensions=768)
    
    def forward(self, x):
        mask = (x != 0).float()
        out, pooler_out = self.longformer(input_ids = x, attention_mask = mask)
        rnn_out = self.rnn(out)
        selected_out = torch.index_select(rnn_out, 1, self.select_idx)
        # embedding = self.conv1d(out.permute(0, 2, 1)).permute(0, 2, 1)
        # embedding = self.attention(self.embedding, self.embedding)
        # return embedding
        return selected_out