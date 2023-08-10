import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# def criterion(yhat, y):
#     out = - 1 * torch.mean(y*torch.log(yhat)+(1-y)*torch.log(1-yhat))
#     return out

# nn.MSELoss() # mean squared error
criterion = nn.BCELoss() # binary cross entropy

class Data(Dataset):
    # def __init__(self, transform=None, train=True):
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

class logistic_reg(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(logistic_reg,self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

if '__main__' == __name__:
    dataset = Data()
    trainloader = DataLoader(dataset=dataset, batch_size=1)
    
    model = logistic_reg(1,1)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        total_loss=0
        
        for x,y in trainloader:
            yhat = model(x)
            loss = criterion(yhat, y)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('epoch: {}, loss: {}'.format(epoch,  total_loss))