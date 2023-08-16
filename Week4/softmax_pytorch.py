import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

#Model
class SoftMax(nn.Module):
    def __init__(self, in_size, out_size):
        super(SoftMax, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        
    def forward(self, x):
        out = self.linear(x)
        return out
    
if __name__ == "__main__":
    #Loss
    criterion = nn.CrossEntropyLoss()

    #Load Data
    data_path= '/Users/angel_de_paula/repos/Deep-Neural-Networks-with-Pytorch/data' 
    train_dataset = dsets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validation_dataset = dsets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    #Data Loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=5000)

    #Model parameters
    _, dim_height, dim_width = train_dataset[0][0].size()
    input_dim = dim_height * dim_width
    output_dim = len(train_dataset.classes)

    model = SoftMax(input_dim, output_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # train parameters
    n_epochs = 10 #100
    accuracy_list = []

    # training Loop
    for epoch in range(n_epochs):
        for x,y in train_loader:
            optimizer.zero_grad()
            z = model(x.view(-1, input_dim))
            loss = criterion(z,y)
            loss.backward()
            optimizer.step()
            
        correct = 0
        
        for x_test, y_test in validation_loader:
            z = model(x_test.view(-1, input_dim))
            _, yhat = torch.max(z.data, 1)
            correct = correct + (yhat == y_test).sum().item()
        accuracy = correct / len(validation_dataset)
        accuracy_list.append(accuracy)
        
        print('Epoch: {} - Loss: {} - Accuracy: {}'.format(epoch, loss.item(), accuracy))