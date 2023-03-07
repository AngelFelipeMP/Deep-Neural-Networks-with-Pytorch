import torch
import matplotlib.pyplot as plt

w = torch.tensor(10.0, requires_grad=True)
X = torch.arange(-3,3,0.1).view(-1,1)
f = -3*X

# print(X)
# print(X.size())
# print(X.ndim)
# print(X.shape)

plt.plot(X.numpy(),f.numpy())
plt.show()

