import torch
from torch.nn import Linear
from torch import nn

class RL(nn.Module):
    def __init__(self, in_size, out_size):
        super(RL,self).__init__()
        self.lr = nn.Linear(in_size, out_size)
        
    def forward(self, x):
        out = self.lr(x)
        return out
    
if '__main__' == __name__:

    torch.manual_seed(1)
    model = Linear(in_features=2, out_features=1)

    print('#### Using pytorch lienar function: ####')
    x = torch.tensor([[1.0,3.0]])
    yhat = model(x)
    print(yhat.item())

    X = torch.tensor([[1.0,1.0],[1.0,2.0],[1.0,3.0]])
    Yhat = model(X)
    print(Yhat.tolist())
        
    print('\n#### Using pytorch custom linear Model: ####')
    torch.manual_seed(1)
    model_custom = RL(2,1)
    yhat = model_custom(x)
    print(yhat.item())

    Yhat = model_custom(X)
    print(Yhat.tolist())