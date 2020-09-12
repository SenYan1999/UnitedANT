import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from apex import amp
from .AudioEncoder import MockingjayEncoder
from .TextEncoder import LongformerEncoder
from .Attention import Attention

def load_audio_data(audio_files):
    audios = []
    for audio in audio_files:
        audios.append(torch.FloatTensor(np.load(audio)))

    lengths = [a.shape[0] for a in audios]
    # max_len = max(lengths)
    max_len = 2 ** 17

    for i in range(len(audios)):
        audio = audios[i]
        if max_len > audio.shape[0]:
            pad = torch.zeros(max_len - audio.shape[0], audio.shape[1])
            audio = torch.cat([audio, pad], dim=0)
        else:
            audio = audio[:max_len, :]
        audios[i] = audio.unsqueeze(dim=0)
    
    audios = torch.cat(audios, dim=0)

    return audios

class DistributedMultimodalityModel(nn.Module):
    def __init__(self, longformer_name, audio_options, table_in, table_out, device):
        super(DistributedMultimodalityModel, self).__init__()
        # state
        self.device = device

        # layers
        self.text_encoder = LongformerEncoder(longformer_name).to('cuda:0')
        self.audio_encoder = MockingjayEncoder(audio_options).to('cuda:1')
        self.table_encoder = nn.Linear(table_in, table_out).to('cuda:0')

        self.attention = Attention(768).to('cuda:0')

        concat_dim = 768 * table_out
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(),
            nn.Linear(concat_dim // 2, concat_dim // 4),
            nn.ReLU(),
            nn.Linear(concat_dim // 4, 3)
        ).to('cuda:0')
    
    def forward(self, text_input, audio_input, tabular_input):
        # text embedding
        text_embedding = self.text_encoder(text_input.to('cuda:0'))

        # audio embedding
        audio_embedding = self.audio_encoder(audio_input.to('cuda:1'))

        # tabular embedding 
        table_embedding = self.table_encoder(tabular_input.to('cuda:0'))

        # concat text, audio and tabular embedding
        concat_text_audio = self.attention(text_embedding, audio_embedding.to('cuda:0'))
        pool_dim = concat_text_audio.shape[1]
        concat_text_audio = F.avg_pool1d(concat_text_audio.permute(0, 2, 1), kernel_size=pool_dim).squeeze(dim=-1)
        concat_embedding = torch.cat([concat_text_audio, table_embedding], dim=-1)

        # out
        pred = self.mlp(concat_embedding)

        return pred
    
    def update(self, batch, optimizer, fp16=False, dist=None):
        # build the batch
        audio_files = [audio for audio in batch[2]]
        audios_input = load_audio_data(audio_files)

        batch_input = {'text_input': batch[1],
                    'audio_input': audios_input,
                    'table_input': batch[3],
                    'true_three': batch[4].to('cuda:0'),
                    'true_seven': batch[5].to('cuda:0'),
                    'true_thirty': batch[6].to('cuda:0')}

        pred = self(batch_input['text_input'], batch_input['audio_input'], batch_input['table_input'])

        loss_three, loss_seven, loss_thirty = F.mse_loss(pred[:, 0], batch_input['true_three']), \
                                              F.mse_loss(pred[:, 1], batch_input['true_seven']), \
                                              F.mse_loss(pred[:, 2], batch_input['true_thirty'])
        loss_total = loss_three + loss_seven + loss_thirty

        if fp16:
            with amp.scale_loss(loss_total, optimizer) as loss:
                loss.backward()
        else:
            loss_total.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        return loss_three.item(), loss_seven.item(), loss_thirty.item()

    def evaluate(self, dev_dataloader):
        mse_three, mse_seven, mse_thirty = [], [], []
        for batch in dev_dataloader:
            # build the batch
            audio_files = [audio for audio in batch[2]]
            audios_input = load_audio_data(audio_files)

            batch_input = {'text_input': batch[1].to(self.device),
                           'audio_input': audios_input.to(self.device),
                           'table_input': batch[3].to(self.device),
                           'true_three': batch[4].to(self.device),
                           'true_seven': batch[5].to(self.device),
                           'true_thirty': batch[6].to(self.device)}

            with torch.no_grad():
                pred = self(batch_input['text_input'], batch_input['audio_input'], batch_input['table_input'])

            loss_three, loss_seven, loss_thirty = F.mse_loss(pred[:, 0], batch_input['true_three']), \
                                                F.mse_loss(pred[:, 1], batch_input['true_seven']), \
                                                F.mse_loss(pred[:, 2], batch_input['true_thirty'])
            mse_three.append(loss_three.item())
            mse_seven.append(loss_seven.item())
            mse_thirty.append(loss_thirty.item())

        return mse_three, mse_seven, mse_thirty
        
