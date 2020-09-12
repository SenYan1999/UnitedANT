import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from apex import amp
from .Attention import Attention
from transformers import AutoModel, AutoConfig
from transformer.nn_transformer import TRANSFORMER

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

class LongformerEncoder(nn.Module):
    def __init__(self, model_name):
        super(LongformerEncoder, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.gradient_checkpointing = True
        self.longformer = AutoModel.from_pretrained(model_name, config=self.config)
    
    def forward(self, x):
        mask = (x != 0).float()
        out, pooler_out = self.longformer(input_ids = x, attention_mask = mask)
        return pooler_out


class MockingjayEncoder(nn.Module):
    def __init__(self, options):
        super(MockingjayEncoder, self).__init__()
        self.mockingjay = TRANSFORMER(options=options, inp_dim=160)
        self.mockingjay.permute_input = True
        self.mockingjay.max_input_length = 1024

    def forward(self, audio):

        # NOTE: there may exist some questions when apply lstm before select because too little parameters solve too large data

        # pretrained model
        reps = self.mockingjay(audio)
        reps = reps[:, -1, :]

        return reps

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

        concat_dim = 768 * 2 + table_out
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(),
            nn.Linear(concat_dim // 2, concat_dim // 4),
            nn.ReLU(),
            nn.Linear(concat_dim // 4, 3)
        )
    
    def forward(self, text_input, audio_input, tabular_input):
        # text embedding
        text_embedding = self.text_encoder(text_input)

        # audio embedding
        audio_embedding = self.audio_encoder(audio_input)

        # tabular embedding 
        table_embedding = self.table_encoder(tabular_input)

        # concat text, audio and tabular embedding
        concat_embedding = torch.cat([text_embedding, audio_embedding, table_embedding], dim=-1)
        
        # out
        pred = self.mlp(concat_embedding)

        return pred
    
    def update(self, batch, optimizer, fp16=False, dist=None):
        # build the batch
        audio_files = [audio for audio in batch[2]]
        audios_input = load_audio_data(audio_files)

        if not dist:
            batch_input = {'text_input': batch[1].cuda(),
                        'audio_input': audios_input.cuda(),
                        'table_input': batch[3].cuda(),
                        'true_three': batch[4].cuda(),
                        'true_seven': batch[5].cuda(),
                        'true_thirty': batch[6].cuda()}
        else:
            batch_input = {'text_input': batch[1].cuda(non_blocking=True),
                        'audio_input': audios_input.cuda(non_blocking=True),
                        'table_input': batch[3].cuda(non_blocking=True),
                        'true_three': batch[4].cuda(non_blocking=True),
                        'true_seven': batch[5].cuda(non_blocking=True),
                        'true_thirty': batch[6].cuda(non_blocking=True)}

        pred = self(batch_input['text_input'], batch_input['audio_input'], batch_input['table_input'])

        loss_three, loss_seven, loss_thirty = F.mse_loss(pred[:, 0], batch_input['true_three']), \
                                              F.mse_loss(pred[:, 1], batch_input['true_seven']), \
                                              F.mse_loss(pred[:, 2], batch_input['true_thirty'])
        loss_total = loss_three + loss_seven + loss_thirty

        if fp16:
            with amp.scale_loss(loss_total, optimizer) as loss:
                loss.backward()
        else:
            # loss_total.backward()
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
        
