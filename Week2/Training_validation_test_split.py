import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class Data(Dataset):
    def __init__(self, transform=None, train=True):
        super(Data,self).__init__()
        self.transform = transform
        self.X = torch.arange(-3,3,0.1).view(-1,1)
        self.Y = -3 * self.X + 0.1 * torch.randn(self.X.size())
        self.len = self.X.shape[0]
        if train:
            self.Y[0]=0
            self.Y[50:55]=20
        else:
            pass
        
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
    
    
class train():
    def __init__(self, trainloader, testloader, lr, epochs):
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.lr = lr
        self.LOSS_train = np.zeros((len(self.lr),self.epochs))
        self.LOSS_test = np.zeros((len(self.lr),self.epochs))
        self.MODELS = []
        
        
    def run(self, criterion = nn.MSELoss()):
        self.criterion = criterion
        self.loss_lr_train = []
        self.loss_lr_test = []
    
        for i,lr in enumerate(self.lr):
            torch.manual_seed(7)
            model = LR(1,1)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            
            for epoch in range(self.epochs):
                train_loss = 0
                test_loss = 0
                
                # Training
                model.train()  
                for x,y in self.trainloader:
                    yhat = model(x)
                    loss = criterion(yhat, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()/len(trainloader)
                self.LOSS_train[i][epoch]=train_loss
                
                # Evaluation
                with torch.no_grad():
                    model.eval()  
                    for x,y in self.testloader:
                        yhat = model(x)
                        loss = criterion(yhat, y)
                        
                        test_loss += loss.item()/len(testloader)
                self.LOSS_test[i][epoch] = test_loss
                
            self.loss_lr_train.append(train_loss)
            self.loss_lr_test.append(test_loss)
                
    def plot_lr(self):
        plt.semilogx(np.array(self.lr), np.array(self.loss_lr_train), label="Training loss")
        plt.semilogx(np.array(self.lr), np.array(self.loss_lr_test), label="Test loss")
        leg = plt.legend(loc='upper center')
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.show()
        
    def plot_loss(self):
        list_epoch = np.arange(1,self.epochs+1)

        for i,lr in enumerate(self.lr):
            plt.plot(list_epoch,self.LOSS_train[i], label="Training loss, lr = {}".format(lr))
        for i,lr in enumerate(self.lr):
            plt.plot(list_epoch,self.LOSS_test[i], label="Test loss, lr = {}".format(lr))
        plt.legend(loc='best')
        plt.ylabel("Loss")
        plt.show()
        

if '__main__' == __name__:
    train_data = Data(train=True)
    test_data = Data(train=False)
    
    trainloader = DataLoader(dataset=train_data, batch_size=1)
    testloader = DataLoader(dataset=test_data, batch_size=1)
    epochs=10
    lr=[0.0001, 0.001, 0.01, 0.1]

    training = train(trainloader,testloader, lr, epochs)
    training.run()
    training.plot_lr()
    training.plot_loss()
    