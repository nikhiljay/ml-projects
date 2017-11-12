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

# seed random numbers to make calculation deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
w = 2 * np.random.random((3, 1)) - 1

for i in range(10000):

    # forward propagation
    output = sigmoid(np.dot(x, w)) # output = sigmoid(input*weight)

    # cost function comparing to desired output
    output_error = y - output

    # multiply how much we missed by the slope of the sigmoid at the values in l1
    output_delta = output_error * sigmoid(output, True)

    # update weights
    w += np.dot(x.T, output_delta)

print("Output After Training:")
print(output)
