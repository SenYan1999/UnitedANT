import torch
import torch.nn as nn

from transformers import AutoModel

class LongformerEncoder(nn.Module):
    def __init__(self, model_name):
        super(LongformerEncoder, self).__init__()
        self.longformer = AutoModel.from_pretrained(model_name)
    
    def forward(self, x):
        out, pooler_out = self.longformer(input_ids = x)
        return out, pooler_out