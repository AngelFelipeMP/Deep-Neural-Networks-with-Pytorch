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
    
    
class LR(nn.Module):
    def __init__(self, in_size, out_size):
        super(LR,self).__init__()
        self.lr = nn.Linear(in_size, out_size)
        
    def forward(self, x):
        return self.lr(x)
    
    
def training(trainloader=None, lr=0.01):
    torch.manual_seed(7)
    model = LR(1,1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    LOSS = []
    epochs = []
    
    for epoch in range(20):
        total_loss = 0
        
        if epoch == 0:
            model.eval()
            with torch.no_grad():
                for x,y in trainloader:
                    yhat = model(x)
                    loss = criterion(yhat, y)
                    total_loss += loss.item()/len(trainloader)
        else:
            model.train()  
            for x,y in trainloader:
                yhat = model(x)
                loss = criterion(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                total_loss += loss.item()/len(trainloader)
                
        LOSS.append(total_loss)
        epochs.append(epoch)
        
    return epochs, LOSS

if '__main__' == __name__:
    dataset = Data()
    trainloader5 = DataLoader(dataset=dataset, batch_size=5)
    trainloader20 = DataLoader(dataset=dataset, batch_size=20)

    epoch5, LOSS5 = training(trainloader=trainloader5)
    epoch20, LOSS20 = training(trainloader=trainloader20)
    
    print(LOSS5)
    print(LOSS20)

    # Loss plot
    plt.plot(epoch5,LOSS5, label="Batch size = 5")
    plt.plot(epoch20,LOSS20, label="Batch size = 20")
    leg = plt.legend(loc='upper center')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()