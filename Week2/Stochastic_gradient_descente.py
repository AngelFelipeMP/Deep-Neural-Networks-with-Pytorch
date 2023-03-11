import torch
import numpy as np
import matplotlib.pyplot as plt

def forward(x):
    y = w*x + b
    return y

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)

if __name__ == '__main__':
    w = torch.tensor(-15.0, requires_grad=True)
    b = torch.tensor(-10.0, requires_grad=True)
    X = torch.arange(-3,3,0.1).view(-1,1)

    f = -3*X
    Y = f + 0.1 * torch.randn(X.size())

    plt.plot(X.numpy(),f.detach().numpy())
    plt.plot(X.numpy(),Y.detach().numpy(),'ro')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    lr = 0.1
    LOSS = []
    epochs = []
    for epoch in range(6):
        total_loss = 0
        for x,y in zip(X,Y):
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