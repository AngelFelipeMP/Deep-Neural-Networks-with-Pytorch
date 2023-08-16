import torch
import torch.nn as nn
from torch import sigmoid
from torch.utils.data import Dataset, DataLoader

#Data
class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-20,20,1).view(-1,1).type(torch.FloatTensor)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x[:,0]>-4) & (self.x[:,0]<4)] = 1.0
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.y.size(0)

#Model
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in,H)
        self.Linear2 = nn.Linear(H, D_out)
        
    def forward(self,x):
        x = sigmoid(self.linear1(x))
        x = sigmoid(self.Linear2(x))
        return x
    
#Train loop
def train(trainLoader, model, optimizer, criterion, epochs=1000):
    cost = []
    total = 0
    for epoch in range(epochs):
        total = 0
        for x,y in trainLoader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat,y.view(-1,1))
            loss.backward()
            optimizer.step()
            
            #acumulative loss
            total+=loss.item()
        cost.append(total)
        
        if epoch % 100 == 0:
            print('Epoch: {} - Loss: {}'.format(epoch, total/len(trainLoader)))
        
    return cost

if __name__ == "__main__":
    #Data 
    dataset = Data()
    TrainLoader = DataLoader(dataset=dataset, batch_size=4)

    model = Net(1,2,1)   

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    cost = train(TrainLoader, model, optimizer, criterion, epochs=1000)