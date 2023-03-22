import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class MLR(nn.Nodule):
    def __init__(self, input_size, output_size):
        super(MLR,self).__init__()
        self.linear = nn.linear(input_size, output_size)
        
    def forward(self, x):
        out = self.linear(x)
        return out
    

class Data2D(Dataset):
    def __init__(self, transform=None):
        self.x = torch.zeros(20,2)
        self.x[:,0] = torch.arange(-1,1,0.1)
        self.x[:,1] = torch.arange(-1,1,0.1)
        self.w = torch.tensor([[1.0],[1.0]])
        self.b = 1
        self.f = torch.mm(self.x, self.w) + self.b
        self.y = self.f * 0.1*torch.randn(self.f.shape)
        # self.y = self.f * 0.1*torch.randn((self.f.shape[0],1))
        self.len = self.x.shape[0]
        self.transform = transform
    
    def __getitem__(self, index):
        if self.transform:
            self.x[index], self.y[index] = self.transform(self.x[index], self.y[index])

        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.len

if __name__ == '__main__':