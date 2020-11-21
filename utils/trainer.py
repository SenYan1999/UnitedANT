import torch
import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm

class Trainer:
    def __init__(self, train_dataloader, dev_dataloader, model, optimizer, device, fp16=False):
        self.train_data = train_dataloader
        self.dev_data = dev_dataloader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.fp16 = fp16

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self, epoch):
        print('Epoch: %2d: Training Model...' % epoch)
        self.model.train()

        total_step = len(self.train_data)
        mses = [[], [], []]
        for step, batch in enumerate(self.train_data):
            # process batch
            mse = self.model.update(batch, self.optimizer, self.fp16)
            mses[0].append(mse[0])
            mses[1].append(mse[1])
            mses[2].append(mse[2])

            if step == 0 or total_step % step == 0:
                print(f'Epoch: {epoch} | 3 MSE: {mse[0]} | 7 MSE: {mse[1]} | 30 MSE: {mse[2]}')

        print(f'Epoch: {epoch} | 3 MSE: {np.mean(mses[0])} | 7 MSE: {np.mean(mses[1])} | 30 MSE: {np.mean(mses[2])}')

    def evaluate_epoch(self, epoch):
        # step1: eval p model
        print('Epoch %2d: Evaluating Model...' % epoch)
        self.model.eval()

        mse_three, mse_seven, mse_thirty = self.model.evaluate(self.dev_data)
        print(f'Epoch: {epoch} | 3 MSE: {np.mean(mse_three)} | 7 MSE: {np.mean(mse_seven)} | 30 MSE: {np.mean(mse_thirty)}')


    def train(self, num_epoch, save_path):
        for epoch in range(num_epoch):
            self.train_epoch(epoch)
            self.evaluate_epoch(epoch)

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