import torch
import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm

class Trainer:
    def __init__(self, train_dataloader, dev_dataloader, model, optimizer, device, tau, fp16=False):
        self.train_data = train_dataloader
        self.dev_data = dev_dataloader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.fp16 = fp16

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tau = tau
        self.model.to(self.device)

    def train_epoch(self, epoch):
        print('Epoch: %2d: Training Model...' % epoch)
        self.model.train()

        total_step = len(self.train_data)
        all_mse = []
        pbar = tqdm(total=len(self.train_data))
        for step, batch in enumerate(self.train_data):
            # process batch
            mse = self.model.update(batch, self.optimizer, self.tau, self.fp16)
            all_mse.append(mse)

            pbar.set_description(f'Epoch: {epoch} | {self.tau} MSE: {mse: 1.2f}')
            pbar.update(1)
        pbar.close()

        print('-------------------------------TRAIN RESULT-------------------------------')
        print(f'Epoch: {epoch} | {self.tau} MSE: {np.mean(all_mse):1.2f}')

    def evaluate_epoch(self, epoch):
        # step1: eval p model
        print('Epoch %2d: Evaluating Model...' % epoch)
        self.model.eval()

        mse = self.model.evaluate(self.dev_data, self.tau)
        print('-------------------------------Evaluate RESULT-------------------------------')
        print(f'Epoch: {epoch} | {self.tau} MSE: {np.mean(mse):1.2f}')

        return mse


    def train(self, num_epoch, save_path):
        best_mse = 1e10
        for epoch in range(num_epoch):
            self.train_epoch(epoch)
            mse = self.evaluate_epoch(epoch)

            best_mse = min(best_mse, mse)
            print('Now best mse is %.2f' % best_mse)

            # save state dict
            path = os.path.join(save_path, 'state_%d_epoch.pt' % epoch)
            self.save_dict(path)

    def save_dict(self, save_path):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state_dict, save_path)

    def load_dict(self, path):
        state_dict = torch.load(path)

        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
