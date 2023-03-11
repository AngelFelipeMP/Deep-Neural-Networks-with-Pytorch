import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

def forward(x):
    y = w*x + b
    return y

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)

class Data(Dataset):
    def __init__(self, transform=None):
        super(Data, self).__init__()
        self.X = torch.arange(-3,3,0.1).view(-1,1)
        self.f = -3 * self.X
        self.Y = self.f + 0.1* torch.randn(self.X.size())
        self.len = self.X.shape[0]
        self.transform = transform
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if self.transform:
            self.X[index], self.Y[index] = self.transform(self.X[index], self.Y[index])

        return self.X[index], self.Y[index]

if __name__ == "__main__":
    w = torch.tensor(-15.0, requires_grad=True)
    b = torch.tensor(-10.0, requires_grad=True)
    
    dataset = Data()
    trainloader = DataLoader(dataset=dataset, batch_size=1)

    lr = 0.1
    LOSS = []
    epochs = []
    for epoch in range(6):
        total_loss = 0
        for x,y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            loss.backward()
            
            w.data = w.data - lr * w.grad.data
            w.grad.data.zero_()
            b.data = b.data - lr * b.grad.data
            b.grad.data.zero_()
            
            total_loss += loss.item()
            
        LOSS.append(total_loss)
        epochs.append(epoch)

    # Loss plot
    plt.plot(epochs,LOSS)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()