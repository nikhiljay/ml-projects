import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

N = 64 # batch size
D_in = 1000 # input dimensions
H = 100 # hidden dimensions
D_out = 1 # output dimensions
learning_rate = 1e-6 # learning rate

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

for i in range(500):
  # forward propagation
  y_pred = x.mm(w1).mm(w2)

  # loss function comparing y_pred to desired output
  loss = (y_pred - y).pow(2).sum()

  print("Epoch:", i, " Loss:", loss.data[0]/100)

  # backprop
  loss.backward()

  # update weights using gradient descent
  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data

  # zero the gradients after backward pass
  w1.grad.data.zero_()
  w2.grad.data.zero_()
