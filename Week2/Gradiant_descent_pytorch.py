import torch
import matplotlib.pyplot as plt

w = torch.tensor(10.0, requires_grad=True)
X = torch.arange(-3,3,0.1).view(-1,1)
f = -3*X  #line equation b=0
Y = f + 0.5* torch.randn(X.size())


plt.plot(X.numpy(),f.numpy())
plt.plot(X.numpy(),Y.numpy(),'ro')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


def forward(x):
  return w*x


def criterion(yhat,y):
  return torch.mean((yhat-y)**2)


lr=0.1
cost=[]
epochs=[]
for epoch in range(4):
  Yhat = forward(X)
  loss = criterion(Yhat, Y)
  loss.backward()
  w.data = w.data - lr * w.grad.data
  w.grad.data.zero_()
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
