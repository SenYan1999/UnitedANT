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
        ids, sents, audios, tables, y_three, y_seven, y_thirty = [], [], [], [], [], [], []

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
                data_slice = self.table[(self.table['PERMNO'] == idx) & (self.table['date'] == int(date))]
                three_days, seven_days, thirty_days = float(data_slice['after_3_days_vol']), \
                    float(data_slice['after_7_days_vol']), float(data_slice['after_30_days_vol'])
                all_ten_retx = []
                for i in range(1, 11):
                    all_ten_retx.append(float(data_slice[f'retx_before_{i}']))
            except:
                print(f'No table data for corp: {corp} and idx: {idx}')
                continue

            ids.append(idx)
            sents.append(doc_idx)
            audios.append(audio)
            tables.append(all_ten_retx)
            y_three.append(three_days)
            y_seven.append(seven_days)
            y_thirty.append(thirty_days)
        
        ids = torch.LongTensor(ids)
        sents = torch.LongTensor(sents)
        tables = torch.FloatTensor(tables)
        y_three = torch.FloatTensor(y_three)
        y_seven = torch.FloatTensor(y_seven)
        y_thirty = torch.FloatTensor(y_thirty)

        return (ids, sents, audios, tables, y_three, y_seven, y_thirty)

    def __getitem__(self, index):
        out = ()
        for item in self.data:
            out += (item[index], )
        return out
    
    def __len__(self):
        return len(self.data)
