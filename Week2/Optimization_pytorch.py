import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class Data(Dataset):
    def __init__(self, transform=None):
        super(Data,self).__init__()
        self.transform = transform
        self.X = torch.arange(-3,3,0.1).view(-1,1)
        self.Y = -3 * self.X + 0.1 * torch.randn(self.X.size())
        self.len = self.X.shape[0]
        
    def __getitem__(self, index):
        if self.transform:
            self.X[index], self.Y[index] = self.transform(self.X[index], self.Y[index])
        return self.X[index], self.Y[index]
        
    def __len__(self):
        return self.len