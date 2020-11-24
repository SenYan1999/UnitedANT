import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from apex import amp
except:
    pass
from .AudioEncoder import BiLSTMEncoder
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
        if max_len - audio.shape[0] > 0:
            pad = torch.zeros(max_len - audio.shape[0], audio.shape[1])
            audio = torch.cat([audio, pad], dim=0)
        else:
            audio = audio[:max_len, :]
        audios[i] = audio.unsqueeze(dim=0)
    
    audios = torch.cat(audios, dim=0)

    return audios

tau2input = {3: 3, 7: 4, 15: 5, 30: 6}
tau2pred = {3: 7, 7: 8, 15: 9, 30: 10}

class MultimodalityModel(nn.Module):
    def __init__(self, longformer_name, audio_dim, hidden_size, num_layers, n_head, table_in, table_out, drop_out, device):
        super(MultimodalityModel, self).__init__()
        # state
        self.device = device

        # layers
        self.text_encoder = LongformerEncoder(longformer_name, n_head=n_head, d_model=768, p_dropout=drop_out)
        self.audio_encoder = BiLSTMEncoder(audio_dim, hidden_size, num_layers, dropout=drop_out, atten_head=n_head)
        self.table_encoder = nn.Linear(table_in, table_out)

        concat_dim = 768 + 768 + table_out
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(),
            nn.Linear(concat_dim // 2, concat_dim // 4),
            nn.ReLU(),
            nn.Linear(concat_dim // 4, 1)
        )
    
    def forward(self, text_input, audio_input, tabular_input):
        # text embedding
        text_embedding = self.text_encoder(text_input)

        # audio embedding
        audio_embedding = self.audio_encoder(audio_input)

        # tabular embedding 
        table_embedding = self.table_encoder(tabular_input)

        # concat text, audio and tabular embedding
        concat_embedding = torch.cat((text_embedding, audio_embedding, table_embedding), dim=-1)

        # out
        pred = self.mlp(concat_embedding)

        return pred
    
    def update(self, batch, optimizer, tau=3, fp16=False):
        # build the batch
        audio_files = [audio for audio in batch[2]]
        audios_input = load_audio_data(audio_files)

        batch_input = {'text_input': batch[1].to(self.device),
                       'audio_input': audios_input.to(self.device),
                       'table_input': batch[tau2input[tau]].unsqueeze(dim=1).to(self.device),
                       'table_pred': batch[tau2pred[tau]].unsqueeze(dim=1).to(self.device)}


        pred = self(batch_input['text_input'], batch_input['audio_input'], batch_input['table_input'])

        loss = F.mse_loss(pred, batch_input['table_pred'])

        if fp16:
            try:
                with amp.scale_loss(loss, optimizer) as loss:
                    loss.backward(retain_graph=True)
            except:
                pass
        else:
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    def evaluate(self, dev_dataloader, tau=3):
        mse = []
        for batch in dev_dataloader:
            # build the batch
            audio_files = [b for b in batch[2]]
            audios_input = load_audio_data(audio_files)

            batch_input = {'text_input': batch[1].to(self.device),
                           'audio_input': audios_input.to(self.device),
                           'table_input': batch[tau2input[tau]].unsqueeze(dim=1).to(self.device),
                           'table_pred': batch[tau2pred[tau]].unsqueeze(dim=1).to(self.device)}

            pred = self(batch_input['text_input'], batch_input['audio_input'], batch_input['table_input'])

            loss = F.mse_loss(pred, batch_input['table_pred'])
            mse.append(loss.item())

        return mse
        
