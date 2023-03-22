import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class MLR(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLR,self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
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
        self.len = self.x.shape[0]
        self.transform = transform
    
    def __getitem__(self, index):
        if self.transform:
            self.x[index], self.y[index] = self.transform(self.x[index], self.y[index])

        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.len

def traing_loop(dataloader, model, loss, optimizer, epochs):
    
    for epoch in range(epochs):
        epoch_loss = 0
        for x,y in dataloader:
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss =+ loss.item()
            
        if epoch % 10 == 0:
            print('Epoch: ', epoch, 'Loss: ', epoch_loss/len(dataloader))

if __name__ == '__main__':
    torch.manual_seed(7)
    dataset = Data2D()
    TrainLoader = DataLoader(dataset=dataset, batch_size=5)
    model = MLR(dataset.x.shape[1], dataset.y.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    traing_loop(TrainLoader, model, criterion, optimizer, 100)