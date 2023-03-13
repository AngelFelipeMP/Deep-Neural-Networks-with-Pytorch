import torch
from torch.utils.data import Dataset, DataLoader
# from SGD_plus_Dataset_DataLoader import forward, criterion, Data
import numpy as np
import matplotlib.pyplot as plt

def forward(x, w, b):
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

def training(trainloader=None, lr = 0.1):
    w = torch.tensor(-15.0, requires_grad=True)
    b = torch.tensor(-10.0, requires_grad=True)
    
    LOSS = []
    epochs = []
    for epoch in range(10):
        total_loss = 0
        
        if epoch == 0:
            print('*************')
            with torch.no_grad():
                for x,y in trainloader:
                    yhat = forward(x, w, b)
                    loss = criterion(yhat, y)
                    total_loss += loss.item()/len(trainloader)
        else:
            for x,y in trainloader:
                yhat = forward(x, w, b)
                loss = criterion(yhat, y)
                loss.backward()
                
                w.data = w.data - lr * w.grad.data
                w.grad.data.zero_()
                b.data = b.data - lr * b.grad.data
                b.grad.data.zero_()
                
                total_loss += loss.item()/len(trainloader)
            
        LOSS.append(total_loss)
        epochs.append(epoch)
    return LOSS, epochs

if __name__ == "__main__":
    dataset = Data()
    trainloader5 = DataLoader(dataset=dataset, batch_size=5)
    trainloader20 = DataLoader(dataset=dataset, batch_size=20)

    epoch5, LOSS5 = training(trainloader=trainloader5)
    epoch20, LOSS20 = training(trainloader=trainloader20)

    # Loss plot
    plt.plot(LOSS5,epoch5, label="Batch size = 5")
    plt.plot(LOSS20,epoch20, label="Batch size = 20")
    leg = plt.legend(loc='upper center')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()