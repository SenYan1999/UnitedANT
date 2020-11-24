import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LongformerModel, LongformerConfig
from .SelfAttention import MultiHeadAttention

class LongformerEncoder(nn.Module):
    def __init__(self, model_name, n_head=4, d_model=768, p_dropout=0.1):
        super(LongformerEncoder, self).__init__()
        config = LongformerConfig.from_pretrained(model_name)
        config.gradient_checkpointing = True
        self.longformer = LongformerModel.from_pretrained(model_name, config=config)
        self.attn = MultiHeadAttention(n_head, d_model, d_model, d_model, dropout=p_dropout)
    
    def forward(self, x):
        x_mask = (x != 0).unsqueeze(dim=-2)
        with torch.no_grad():
            out_longformer, _ = self.longformer(input_ids = x)
        out_attn = self.attn(out_longformer, out_longformer, out_longformer, mask=x_mask)
        out = F.max_pool1d(out_attn.transpose(-2, -1), kernel_size=out_attn.shape[-2]).squeeze(dim=-1)
        return out