import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self, in_size, out_size):
        super(Softmax, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        
    def forward(self, x):
        out = self.linear(x)
        return out

if __name__ == "__main__":
    torch.manual_seed(1)

    model = Softmax(2,3)
    # x = torch.tensor([[1.0,2.0]])
    x = torch.tensor([[1.0,1.0],[1.0,2.0],[1.0,3.0]])

    z = model(x)

    # yindex,yhat = z.max(1)
    _,yhat = z.max(1)

    print("The result of softmax is ", yhat)