import torch
import os
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer 
from tqdm import tqdm

class MultimodalityDataset(Dataset):
    def __init__(self, company2idx, audio_dir, table, text_dir, text_max_len, audio_max_len, min_date, max_date, longformer_name):
        self.company2idx = company2idx
        self.audio_dir = audio_dir
        self.table = table
        self.text_dir = text_dir
        self.text_max_len = text_max_len
        self.audio_max_len = audio_max_len
        self.tokenizer = AutoTokenizer.from_pretrained(longformer_name)

        self.data = self.process_raw_data(min_date, max_date)

    def process_raw_data(self, min_date, max_date):
        all_corps = os.listdir(self.audio_dir)
        all_corps = [corp[0:-4] for corp in all_corps]
        ids, sents, audios, tables = [], [], [], []
        y_three_before, y_seven_before, y_fifteen_before, y_thirty_before = [], [], [], []
        y_three_after, y_seven_after, y_fifteen_after, y_thirty_after = [], [], [], []

        for corp in tqdm(all_corps):
            data_slice = ()

            # audio and text data
            name, date = corp.replace(',', '').split('_')
            if not(min_date <= int(date) < max_date):
                continue

            try:
                idx = int(self.company2idx[name])
            except:
                idx = int(self.company2idx[name.replace(' Inc', '').replace('.', '')])

            # append audio data
            audio = os.path.join(self.audio_dir, f'{corp}.npy')

            # process text
            try:
                docs = ''
                with open(os.path.join(self.text_dir, f'{corp}.txt'), 'r') as f:
                    for line in f.readlines():
                        docs += line
                doc_tokens = self.tokenizer.tokenize(docs)
                doc_idx = self.tokenizer.convert_tokens_to_ids(doc_tokens)
                if self.text_max_len < len(doc_tokens):
                    doc_idx = doc_idx[0:self.text_max_len]
                else:
                    doc_idx += [0 for i in range(self.text_max_len - len(doc_idx))]
            except:
                print(f'No text data for corp: {corp} and idx: {idx}')
                continue

            # table data
            try:
                date = str(int(str(date)[:4])) + '/' + str(int(str(date)[4:6])) + '/' + str(int(str(date)[6:]))
                data_slice = self.table[(self.table['permno'] == idx) & (self.table['date'] == date)]
                before_three_days, before_seven_days, before_fifteen_days, before_thirty_days = float(data_slice['vol_before3']), \
                    float(data_slice['vol_before7']), float(data_slice['vol_before15']), float(data_slice['vol_before30'])
                after_three_days, after_seven_days, after_fifteen_days, after_thirty_days = float(data_slice['vol_after3']), \
                    float(data_slice['vol_after7']), float(data_slice['vol_after15']), float(data_slice['vol_after30'])
            except:
                print(f'No table data for corp: {corp} and idx: {idx}')
                continue

            ids.append(idx)
            sents.append(doc_idx)
            audios.append(audio)
            y_three_before.append(before_three_days)
            y_seven_before.append(before_seven_days)
            y_fifteen_before.append(before_fifteen_days)
            y_thirty_before.append(before_thirty_days)
            y_three_after.append(after_three_days)
            y_seven_after.append(after_seven_days)
            y_fifteen_after.append(after_fifteen_days)
            y_thirty_after.append(after_thirty_days)
        
        ids = torch.LongTensor(ids)
        sents = torch.LongTensor(sents)
        y_three_before = torch.FloatTensor(y_three_before)
        y_seven_before = torch.FloatTensor(y_seven_before)
        y_fifteen_before = torch.FloatTensor(y_fifteen_before)
        y_thirty_before = torch.FloatTensor(y_thirty_before)
        y_three_after = torch.FloatTensor(y_three_after)
        y_seven_after = torch.FloatTensor(y_seven_after)
        y_fifteen_after = torch.FloatTensor(y_fifteen_after)
        y_thirty_after = torch.FloatTensor(y_thirty_after)

        return (ids, sents, audios, y_three_before, y_seven_before, y_fifteen_before, y_thirty_before, \
            y_three_after, y_seven_after, y_fifteen_after, y_thirty_after)

    def __getitem__(self, index):
        out = ()
        for item in self.data:
            out += (item[index], )
        return out
    
    def __len__(self):
        return self.data[0].shape[0]
