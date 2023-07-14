import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

class logistic_regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(logistic_regression,self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

if __name__ == '__main__':

    z = torch.arange(-100,100,0.1).view(-1,1)

    # sig = nn.Sigmoid() # sigmoid object
    # sig = torch.sigmoid # sigmoid fuction
    # sig = logistic_regression(1,1) # custion fuction
    sig = nn.Sequential(nn.Linear(1,1),nn.Sigmoid()) # sequential object

    yhat = sig(z)

    plt.plot(z.numpy(),yhat.detach().numpy())
    plt.show()
