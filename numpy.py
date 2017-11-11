import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
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

# seed random numbers to make calculation deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
w = 2 * np.random.random((3, 1)) - 1

for iter in xrange(10000):

    # forward propagation
    l0 = x
    l1 = nonlin(np.dot(l0, w))

    l1_error = y - l1 # cost function

    # multiply how much we missed by the slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    w += np.dot(l0.T, l1_delta)

print "Output After Training:" + l1
