import torch
import torch.nn as nn
from .RNN import LSTM
from transformer.nn_transformer import TRANSFORMER
from .Attention import Attention

class MockingjayEncoder(nn.Module):
    def __init__(self, options):
        super(MockingjayEncoder, self).__init__()
        self.mockingjay = TRANSFORMER(options=options, inp_dim=160)
        self.mockingjay.permute_input = True
        self.mockingjay.max_input_length = 1024
        self.select_idx = torch.tensor([i * 512 for i in range(0, 256)]).long()
        self.lstm = LSTM(input_size=768, hidden_size=768 // 2, bidirectional=True, num_layers=1, batch_first=True)

        self.attention = Attention(dimensions=768)
    
    def forward(self, audio, audio_len):

        # NOTE: there may exist some questions when apply lstm before select because too little parameters solve too large data

        # pretrained model
        reps = self.mockingjay(audio)

        # mask reps
        audio_mask = torch.zeros(reps.size(0), reps.size(1)).cuda()
        for i, mask_len in zip(range(audio_mask.size(0)), audio_len):
            audio_mask[i, :mask_len] = 1
        reps = reps * audio_mask.unsqueeze(dim=-1).expand(reps.shape)

        # select specific id and apply lstm
        reps = torch.index_select(reps, 1, self.select_idx.cuda())
        reps = self.lstm(reps)

        # self attention
        self.attention(reps, reps)

        # return embedding
        return reps
