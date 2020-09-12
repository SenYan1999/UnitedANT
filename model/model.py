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
    audios_len = []
    for audio in audio_files:
        audios.append(torch.FloatTensor(np.load(audio)))

    lengths = [a.shape[0] for a in audios]
    # max_len = max(lengths)
    max_len = 2 ** 17

    for i in range(len(audios)):
        audio = audios[i]
        audio_len = audio.size(0) if audio.size(0) < max_len else max_len
        if max_len > audio.shape[0]:
            pad = torch.zeros(max_len - audio.shape[0], audio.shape[1])
            audio = torch.cat([audio, pad], dim=0)
        else:
            audio = audio[:max_len, :]
        audios[i] = audio.unsqueeze(dim=0)
        audios_len.append(audio_len)
    
    audios = torch.cat(audios, dim=0)
    audios_len = torch.LongTensor(audios_len)

    return audios, audios_len

class MultimodalityModel(nn.Module):
    def __init__(self, longformer_name, audio_options, table_in, table_out, device):
        super(MultimodalityModel, self).__init__()
        # state
        self.device = device

        # layers
        self.text_encoder = LongformerEncoder(longformer_name)
        self.audio_encoder = MockingjayEncoder(audio_options)
        self.table_encoder = nn.Linear(table_in, table_out)

        self.attention = Attention(768)

        concat_dim = 768 + table_out
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(),
            nn.Linear(concat_dim // 2, concat_dim // 4),
            nn.ReLU(),
            nn.Linear(concat_dim // 4, 1)
        )
    
    def forward(self, text_input, audio_input, audio_len, tabular_input):
        # audio embedding
        audio_embedding = self.audio_encoder(audio_input, audio_len)

        # text embedding
        text_embedding = self.text_encoder(text_input)

        # tabular embedding 
        table_embedding = self.table_encoder(tabular_input)

        # concat text, audio and tabular embedding
        concat_text_audio = self.attention(text_embedding, audio_embedding)
        pool_dim = concat_text_audio.shape[1]
        concat_text_audio = F.avg_pool1d(concat_text_audio.permute(0, 2, 1), kernel_size=pool_dim).squeeze(dim=-1)
        concat_embedding = torch.cat([concat_text_audio, table_embedding], dim=-1)
        
        # out
        pred = self.mlp(concat_embedding)

        return pred
    
    def update(self, batch, optimizer, fp16=False, dist=None):
        # build the batch
        audio_files = [audio for audio in batch[2]]
        audios_input, audios_len = load_audio_data(audio_files)

        if not dist:
            batch_input = {'text_input': batch[1].cuda(),
                        'audio_input': audios_input.cuda(),
                        'audio_len': audios_len.cuda(),
                        'table_input': batch[3].cuda(),
                        'true_three': batch[4].cuda(),
                        'true_seven': batch[5].cuda(),
                        'true_fifteen': batch[6].cuda(),
                        'true_thirty': batch[6].cuda()}
        else:
            batch_input = {'text_input': batch[1].cuda(non_blocking=True),
                        'audio_input': audios_input.cuda(non_blocking=True),
                        'table_input': batch[3].cuda(non_blocking=True),
                        'true_three': batch[4].cuda(non_blocking=True),
                        'true_seven': batch[5].cuda(non_blocking=True),
                        'true_thirty': batch[6].cuda(non_blocking=True)}

        pred = self(batch_input['text_input'], batch_input['audio_input'], batch_input['audio_len'], batch_input['table_input'])

        loss_three, loss_seven, loss_fifteen, loss_thirty = F.mse_loss(pred[:, 0], batch_input['true_three']), \
                                              F.mse_loss(pred[:, 1], batch_input['true_seven']), \
                                              F.mse_loss(pred[:, 2], batch_input['true_fifteen']), \
                                              F.mse_loss(pred[:, 3], batch_input['true_thirty'])
        loss_total = loss_three + loss_seven + loss_fifteen + loss_thirty

        if fp16:
            with amp.scale_loss(loss_total, optimizer) as loss:
                loss.backward()
        else:
            # loss_total.backward()
            loss_total.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        return loss_three.item(), loss_seven.item(), loss_fifteen.item(), loss_thirty.item()
    
    def update_one(self, batch, optimizer, fp16=False, dist=None):
        # build the batch
        audio_files = [audio for audio in batch[2]]
        audios_input, audios_len = load_audio_data(audio_files)

        if not dist:
            batch_input = {'text_input': batch[1].cuda(),
                        'audio_input': audios_input.cuda(),
                        'audio_len': audios_len.cuda(),
                        'table_input': batch[3].cuda(),
                        'true_three': batch[4].cuda(),
                        'true_seven': batch[5].cuda(),
                        'true_fifteen': batch[6].cuda(),
                        'true_thirty': batch[6].cuda()}
        else:
            batch_input = {'text_input': batch[1].cuda(non_blocking=True),
                        'audio_input': audios_input.cuda(non_blocking=True),
                        'table_input': batch[3].cuda(non_blocking=True),
                        'true_three': batch[4].cuda(non_blocking=True),
                        'true_seven': batch[5].cuda(non_blocking=True),
                        'true_thirty': batch[6].cuda(non_blocking=True)}

        pred = self(batch_input['text_input'], batch_input['audio_input'], batch_input['audio_len'], batch_input['table_input'])

        loss_three = F.mse_loss(pred.squeeze(), batch_input['true_seven'])
        loss_total = loss_three

        if fp16:
            with amp.scale_loss(loss_total, optimizer) as loss:
                loss.backward()
        else:
            # loss_total.backward()
            loss_total.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        return loss_three.item(), 0, 0, 0

    def evaluate(self, dev_dataloader):
        mse_three, mse_seven, mse_fifteen, mse_thirty = [], [], [], []
        for batch in dev_dataloader:
            # build the batch
            audio_files = [audio for audio in batch[2]]
            audios_input, audios_len = load_audio_data(audio_files)

            batch_input = {'text_input': batch[1].to(self.device),
                           'audio_input': audios_input.to(self.device),
                           'audio_len': audios_len.to(self.device),
                           'table_input': batch[3].to(self.device),
                           'true_three': batch[4].to(self.device),
                           'true_seven': batch[5].to(self.device),
                           'true_fifteen': batch[6].to(self.device),
                           'true_thirty': batch[7].to(self.device)}

            with torch.no_grad():
                pred = self(batch_input['text_input'], batch_input['audio_input'], batch_input['audio_len'], batch_input['table_input'])

            # loss_three, loss_seven, loss_fifteen, loss_thirty = F.mse_loss(pred[:, 0], batch_input['true_three']), \
            #                                     F.mse_loss(pred[:, 1], batch_input['true_seven']), \
            #                                     F.mse_loss(pred[:, 2], batch_input['true_fifteen']), \
            #                                     F.mse_loss(pred[:, 3], batch_input['true_thirty'])
            # mse_three.append(loss_three.item())
            # mse_seven.append(loss_seven.item())
            # mse_fifteen.append(loss_fifteen.item())
            # mse_thirty.append(loss_thirty.item())

        # return mse_three, mse_seven, mse_fifteen, mse_thirty

            loss = F.mse_loss(pred.squeeze(), batch_input['true_seven'])
            mse_three.append(loss.item())
            mse_seven.append(0)
            mse_fifteen.append(0)
            mse_thirty.append(0)

        return mse_three, mse_seven, mse_fifteen, mse_thirty
