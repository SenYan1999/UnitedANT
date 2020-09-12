import torch
import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm

class Trainer:
    def __init__(self, train_dataloader, dev_dataloader, model, optimizer, fp16=False):
        self.train_data = train_dataloader
        self.dev_data = dev_dataloader
        self.model = model
        self.optimizer = optimizer
        self.fp16 = fp16

    def train_epoch(self, epoch, dist=False):
        print('Epoch: %2d: Training Model...' % epoch)
        self.model.train()

        total_step = len(self.train_data)
        mses = [[], [], [], []]

        pbar = tqdm(total = total_step)
        for step, batch in enumerate(self.train_data):
            # process batch
            # mse = self.model.update(batch, self.optimizer, self.fp16, dist=dist)
            mse = self.model.update_one(batch, self.optimizer, self.fp16, dist=dist)
            mses[0].append(mse[0])
            mses[1].append(mse[1])
            mses[2].append(mse[2])
            mses[3].append(mse[3])

            pbar.set_description(f'Epoch: {epoch: 2d} | 3 MSE: {mse[0]:.3f} | 7 MSE: {mse[1]:.3f} | 15 MSE: {mse[2]:.3f} | 30 MSE: {mse[3]:.3f}')
            pbar.update(1)

            if step % 50 == 0:
                print(f'Epoch: {epoch: 2d} | Step: [{step: 3d} / {total_step: 3d}] | 3 MSE: {mse[0]:.3f} | 7 MSE: {mse[1]:.3f} | 15 MSE: {mse[2]:.3f} | 30 MSE: {mse[2]:.3f}')
        pbar.close()

        print(f'Epoch: {epoch: 2d} | 3 MSE: {np.mean(mses[0]): .3f} | 7 MSE: {np.mean(mses[1]): .3f} | 15 MSE: {np.mean(mses[2]):.3f} | 30 MSE: {np.mean(mses[3]): .3f}')

    def evaluate_epoch(self, epoch):
        # step1: eval p model
        print('Epoch %2d: Evaluating Model...' % epoch)
        self.model.eval()

        mse_three, mse_seven, mse_fifteen, mse_thirty = self.model.evaluate(self.dev_data)
        print(f'Epoch: {epoch: 2d} | 3 MSE: {np.mean(mse_three): .3f} | 7 MSE: {np.mean(mse_seven): .3f} | 15 MSE: {np.mean(mse_fifteen):.3f} | 30 MSE: {np.mean(mse_thirty): .3f}')


    def train(self, num_epoch, save_path, dist=False):
        for epoch in range(num_epoch):
            self.train_epoch(epoch, dist=dist)
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
