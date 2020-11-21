import torch
import torch.nn as nn
from transformer.nn_transformer import TRANSFORMER

class MockingjayEncoder(nn.Module):
    def __init__(self, options):
        super(MockingjayEncoder, self).__init__()
        self.mockingjay = TRANSFORMER(options=options, inp_dim=160)
        self.mockingjay.permute_input = False
        self.mockingjay.max_input_length = 1024
    
    def forward(self, audio):
        reps = self.mockingjay(audio)
        kernel_size = reps.shape[1]
        return nn.functional.max_pool1d(reps.permute(0, 2, 1), kernel_size=kernel_size).squeeze(dim=-1)
