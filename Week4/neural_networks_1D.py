import torch
import torch.nn as nn
from torch import sigmoid

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
def train(X,Y, model, optimizer, criterion, epochs=1000):
    cost = []
    total = 0
    for epoch in range(epochs):
        total = 0
        for x,y in zip(X,Y):
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat,y.view(-1))
            loss.backward()
            optimizer.step()
            
            #acumulative loss
            total+=loss.item()
        cost.append(total)
        
        if epoch % 100 == 0:
            print('Epoch: {} - Loss: {}'.format(epoch, total/len(X)))
        
    return cost

if __name__ == "__main__":
    #Data
    X = torch.arange(-20,20,1).view(-1,1).type(torch.FloatTensor)
    Y = torch.zeros(X.shape[0])
    Y[(X[:,0]>-4) & (X[:,0]<4)] = 1.0

    model = Net(1,2,1)   
    # model = nn.Sequential(nn.Linear(1, 2), nn.Sigmoid(), nn.Linear(2, 1), nn.Sigmoid())

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    cost = train(X,Y, model, optimizer, criterion, epochs=1000)
    