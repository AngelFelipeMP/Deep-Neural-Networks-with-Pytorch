import torch
import numpy as np
import matplotlib.pyplot as plt


def forward(x):
    y = w*x + b
    return y


def criterion(yhat,y):
    return torch.mean((yhat-y)**2)
        

if __name__ == "__main__":
    w = torch.tensor(-15.0, requires_grad=True)
    b = torch.tensor(-10.0, requires_grad=True)
    X = torch.arange(-3,3,0.1).view(-1,1)
    f = -1*X -1
    Y = f + 0.1* torch.randn(X.size())


    plt.plot(X.numpy(),f.detach().numpy())
    plt.plot(X.numpy(),Y.detach().numpy(),'ro')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()





    lr=0.1
    cost=[]
    epochs=[]

    for epoch in range(6):
        Yhat = forward(X)
        loss = criterion(Yhat, Y)
        
        loss.backward()

        w.data = w.data - lr * w.grad.data
        w.grad.data.zero_()

        b.data = b.data - lr * b.grad.data
        b.grad.data.zero_()

        cost.append(loss.item())
        epochs.append(epoch)

        plt.plot(X.detach().numpy(),Yhat.detach().numpy(), label="line"+str(epoch))
    leg = plt.legend(loc='upper center')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    # Loss plot
    plt.plot(epochs,cost)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()