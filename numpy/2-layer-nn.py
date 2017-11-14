import numpy as np

# sigmoid function
def sigmoid(x, deriv=False):
    if(deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# input dataset
x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

# initialize weights randomly with mean 0
w = 2 * np.random.random((3, 1)) - 1

for i in range(10000):

    # forward propagation
    l0 = x # not being used but is layer 0
    l1 = sigmoid(np.dot(x, w))

    # cost function comparing l1 to desired output
    l1_error = y - l1

    # multiply cost by derivative of activation function
    l1_delta = l1_error * sigmoid(l1, True)

    # update weights
    w += np.dot(x.T, l1_delta)

print("Output After Training:")
print(l1)
