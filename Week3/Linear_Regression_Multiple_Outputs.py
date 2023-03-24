import torch
from torch import nn
from torch.nn import Linear

class MLR(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLR,self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        out = self.linear(x)
        return out

if __name__ == '__main__':
    torch.manual_seed(1)
    
    print('\n================================')
    X = torch.tensor([[1.0,1.0],[1.0,2.0],[1.0,3.0]])
    print(X)
    print(X.shape)
    
    
    model_lr = Linear(2,2)
    print(list(model_lr.parameters()))
    print(model_lr.state_dict())
    Yhat = model_lr(X)
    print(Yhat)

    
    print('\n================================')
    model_custon = MLR(2,2)
    print(list(model_custon.parameters()))
    print(model_custon.state_dict())
    Yhat = model_custon(X)
    print(Yhat)

    
    
    