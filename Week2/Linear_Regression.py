import argparse

def linear_regration_base_pytorch():
    import torch
    '''Linear Regression with PyTorch'''
    w=torch.tensor(2.0, requires_grad=True)
    b=torch.tensor(-1.0, requires_grad=True)

    def forward(x):
        y=w*x+b
        return y

    x=torch.tensor([1.0])
    yhat=forward(x)
    print('X1:', x)
    print('Yhat1:', yhat)
    print('\n')
    
    x=torch.tensor([[1.0],[2.0]])
    yhat=forward(x)
    print('X2:', x)
    print('Yhat2:', yhat)
    
    
def linear_regration_torch_nn():
    '''Linear Regression with PyTorch nn module'''
    import torch
    from torch.nn import Linear
    torch.manual_seed(1)

    model=Linear(in_features=1, out_features=1)
    print('Model Parameters',list(model.parameters()))
    print('\n')

    x=torch.tensor([0.0])
    yhat=model(x)
    print('X1:', x)
    print('Yhat1:', yhat)
    print('\n')
    
    x=torch.tensor([[1.0],[2.0]])
    yhat=model(x)
    print('X2:', x)
    print('Yhat2:', yhat)
    
    #print parameters
    print_model_parameters(model)
    
    
def linear_regration_custon_model():
    '''Linear Regression with Custom Model'''
    import torch
    from torch import nn
    torch.manual_seed(1)

    class LR(nn.Module):
        def __init__(self, input_size, output_size):
            super(LR,self).__init__()
            self.linear=nn.Linear(input_size, output_size)
            
        def forward(self,x):
            out=self.linear(x)
            return out
        
    model=LR(1,1)
    print('Model Parameters',list(model.parameters()))
    print('\n')
    
    x=torch.tensor([0.0])
    yhat=model(x)
    print('X1:', x)
    print('Yhat1:', yhat)
    print('\n')
    
    x=torch.tensor([[1.0],[2.0]])
    yhat=model(x)
    print('X2:', x)
    print('Yhat2:', yhat)
    
    #print parameters
    print_model_parameters(model)
    
def print_model_parameters(model):
    '''Print model parameters'''
    print('\n')
    print('Model Parameters')
    print('Python dictionary:', model.state_dict())
    print('keys:', model.state_dict().keys())
    print('values:', model.state_dict().values())


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="", type=str, help="Linear Regression Pytorch implementation type: base, nn, custom")
    args = parser.parse_args()

    # check information config
    if args.type == '':
        print('Specifying Linear Regression Pytorch implementation type')
        exit(1)
        
    else :
        if args.type == 'base':
            linear_regration_base_pytorch()
        elif args.type == 'nn':
            linear_regration_torch_nn()
        elif args.type == 'custom':
            linear_regration_custon_model()
        else:
            print('Invalid type. Please choose between base, nn, custom')
            exit(1)

