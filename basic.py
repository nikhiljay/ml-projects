import torch
from torch.autograd import Variable
import numpy as np

dtype = torch.FloatTensor # a tensor is essentially an n-dimensional array

n = 64 # batch size
d_in = 1000 # input dimensions
hidden = 100 # number of nodes in hidden layer
d_out = 10 # output dimensions
lr = 1e-6 # learning rate
epochs = 500 # number of epochs/iterations

x = Variable(torch.randn(n, d_in).type(dtype), requires_grad=False) # random tensor to hold inputs
y = Variable(torch.randn(n, d_out).type(dtype), requires_grad=False) # random tensor to hold outputs

# random weight tensors (will use gradient descent)
w1 = Variable(torch.randn(d_in, hidden).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(hidden, d_out).type(dtype), requires_grad=True)

for i in range(epochs):
  output = x.mm(w1).mm(w2) # input tensor multiplied by both weights

  loss = (output - y).pow(2).sum() # loss function comparing experimental output with desired output

  print("Epoch:", i, "  Loss:", loss.data[0]/100)

  loss.backward() # compute gradient of loss on w1 and w2

  # update weights using gradient descent
  w1.data -= lr * w1.grad.data
  w2.data -= lr * w2.grad.data

  # reset gradients after running backward pass
  w1.grad.data.zero_()
  w2.grad.data.zero_()
