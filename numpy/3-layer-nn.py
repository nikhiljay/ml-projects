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
y = np.array([[0, 1, 1, 0]]).T

# initialize weights randomly with mean 0
w0 = 2 * np.random.random((3, 4)) - 1
w1 = 2 * np.random.random((4, 1)) - 1

for i in range(10000):

    # forward propagation
    l0 = x # not being used but is layer 0
    l1 = sigmoid(np.dot(x, w0))
    l2 = sigmoid(np.dot(l1, w1))

    # cost function comparing l2 to desired output
    l2_error = y - l2

    # multiply cost of l2 by derivative of activation function bc thats how backprop works
    l2_delta = l2_error * sigmoid(l2, True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = np.dot(l2_delta, w1.T)

    # multiply cost of l1 by derivative of activation function bc thats how backprop works
    l1_delta = l1_error * sigmoid(l1, True)

    # update weights
    w0 += np.dot(l0.T, l1_delta)
    w1 += np.dot(l1.T, l2_delta)

print("Output After Training:")
print(l2)
