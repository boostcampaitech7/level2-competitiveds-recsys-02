import math
import numpy as np
from tqdm import tqdm
import torch
from torch import nn                         
from torch.utils.data import Dataset     
from torch.optim.lr_scheduler import _LRScheduler

# dataset load
class TrainDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y  

    def __len__(self):
        return self.len

class TestDataset(Dataset):
    
    def __getitem__(self, index):
        x = self.x_data[index]
        return x
    
# model 
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__() # 모델 연산 정의
        self.in_dim = in_dim
        # self.hidden_dim = hidden_dim        
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1, bias=True)
        )
        

    def forward(self, x): # 모델 연산의 순서를 정의
        x = self.mlp(x)
        return x
    

# model train
def train(model, train_loader, valid_loader, criterion, optimizer, epochs, device, checkpoint_name, scheduler = None):

    model.to(device)
    model.train()
    best_loss = np.inf
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            input, target = batch
            input, target = input.to(device), target.to(device)

            output = model(input)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            train_loss += loss.detach().item()
        
        train_loss /= len(train_loader.dataset)
        
        valid_loss = evaluation(model, valid_loader, criterion, device)
        
        if valid_loss < best_loss: 
            torch.save(model.state_dict(), f'checkpoint/{checkpoint_name}_parameters.pt')
            best_loss = valid_loss
       
        if scheduler:
            scheduler.step()
        
        print(f'Epoch: {epoch+1}, Training Loss: {train_loss}, Validation Loss: {valid_loss}')



# model test
def evaluation(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader):

            input, target = batch
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    return test_loss


# schedular
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
