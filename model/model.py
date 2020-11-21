import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from apex import amp
except:
    pass
from .AudioEncoder import MockingjayEncoder
from .TextEncoder import LongformerEncoder
from .SelfAttention import MultiHeadAttention

def load_audio_data(audio_files):
    audios = []
    for audio in audio_files:
        audios.append(torch.FloatTensor(np.load(audio)))

    lengths = [a.shape[0] for a in audios]
    max_len = max(lengths)

    for i in range(len(audios)):
        audio = audios[i]
        assert len(audio.shape) == 2
        pad = torch.zeros(max_len - audio.shape[0], audio.shape[1])
        audio = torch.cat([audio, pad], dim=0)
        audios[i] = audio.unsqueeze(dim=0)
    
    audios = torch.cat(audios, dim=0)

    return audios

class MultimodalityModel(nn.Module):
    def __init__(self, longformer_name, audio_options, table_in, table_out, device):
        super(MultimodalityModel, self).__init__()
        # state
        self.device = device

        # layers
        self.text_encoder = LongformerEncoder(longformer_name)
        self.audio_encoder = MockingjayEncoder(audio_options)
        self.table_encoder = nn.Linear(table_in, table_out)

        concat_dim = 768 + 768 + table_out
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(),
            nn.Linear(concat_dim // 2, concat_dim // 4),
            nn.ReLU(),
            nn.Linear(concat_dim // 4, 3)
        )
    
    def forward(self, text_input, audio_input, tabular_input):
        # text embedding
        _, text_embedding = self.text_encoder(text_input)

        # audio embedding
        audio_embedding = self.audio_encoder(audio_input)

        # tabular embedding 
        table_embedding = self.table_encoder(tabular_input)

        # concat text, audio and tabular embedding
        concat_embedding = torch.cat((text_embedding, audio_embedding, table_embedding), dim=-1)

        # out
        pred = self.mlp(concat_embedding)

        return pred
    
    def update(self, batch, optimizer, fp16=False):
        # build the batch
        audio_files = [audio for audio in batch[2]]
        audios_input = load_audio_data(audio_files)

        batch_input = {'text_input': batch[1].to(self.device),
                       'audio_input': audios_input.to(self.device),
                       'table_input': batch[3].to(self.device),
                       'true_three': batch[4].to(self.device),
                       'true_seven': batch[5].to(self.device),
                       'true_thirty': batch[6].to(self.device)}


        pred = self(batch_input['text_input'], batch_input['audio_input'], batch_input['table_input'])
        print(pred.shape)

        loss_three, loss_seven, loss_thirty = F.mse_loss(pred[:, 0], batch_input['true_three']), \
                                              F.mse_loss(pred[:, 1], batch_input['true_seven']), \
                                              F.mse_loss(pred[:, 2], batch_input['true_thirty'])
        loss_total = loss_three + loss_seven + loss_thirty

        if fp16:
            try:
                with amp.scale_loss(loss_total, optimizer) as loss:
                    loss.backward(retain_graph=True)
            except:
                pass
        else:
            loss_total.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        return loss_three.item(), loss_seven.item(), loss_thirty.item()

    def evaluate(self, dev_dataloader):
        mse_three, mse_seven, mse_thirty = [], [], []
        for batch in dev_dataloader:
            # build the batch
            audio_files = [b[2] for b in batch]
            audios_input = [a.to(self.device) for a in load_audio_data(audio_files)]

            batch_input = {'text_input': batch_input[1].to(self.device),
                           'audio_input': audios_input,
                           'able_input': batch_input[3].to(self.device),
                           'true_three': batch_input[4].to(self.device),
                           'true_seven': batch_input[5].to(self.device),
                           'true_thirty': batch_input[6].to(self.device)}

            pred = self(batch_input['text_input'], batch_input['audio_input'], batch_input['table_input'])

            loss_three, loss_seven, loss_thirty = F.mse_loss(pred[:, 0], batch['true_three']), \
                                                F.mse_loss(pred[:, 1], batch['true_seven']), \
                                                F.mse_loss(pred[:, 2], batch['true_thirty'])
            mse_three.append(loss_three)
            mse_seven.append(loss_seven)
            mse_thirty.append(loss_thirty)

        return mse_three, mse_seven, mse_thirty
        
