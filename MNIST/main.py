from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# total training examples in dataset
def TRAIN_SIZE(num):
    print ('Total Training Images in Dataset = ' + str(mnist.train.images.shape))
    print ('--------------------------------------------------')
    x_train = mnist.train.images[:num,:]
    print ('x_train Examples Loaded = ' + str(x_train.shape))
    y_train = mnist.train.labels[:num,:]
    print ('y_train Examples Loaded = ' + str(y_train.shape))
    print('')
    return x_train, y_train

# total test examples in dataset
def TEST_SIZE(num):
    print ('Total Test Examples in Dataset = ' + str(mnist.test.images.shape))
    print ('--------------------------------------------------')
    x_test = mnist.test.images[:num,:]
    print ('x_test Examples Loaded = ' + str(x_test.shape))
    y_test = mnist.test.labels[:num,:]
    print ('y_test Examples Loaded = ' + str(y_test.shape))
    return x_test, y_test

# show image of a sample digit
def display_digit(num):
    print('Correct:', y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

x = tf.placeholder(tf.float32, shape=[None, 784]) # input
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # actual output
W = tf.Variable(tf.zeros([784,10])) # weight
b = tf.Variable(tf.zeros([10])) # bias
y = tf.nn.softmax(tf.matmul(x,W) + b) # predicited output
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # loss function

x_train, y_train = TRAIN_SIZE(5500) # number of training samples
x_test, y_test = TEST_SIZE(10000) # number of test samples
LEARNING_RATE = 0.05
EPOCHS = 200 # more epochs = more accurate predictions

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy) # perform gradient descent
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # correct prediction is when probability is 1
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train
for i in range(EPOCHS+1):
    sess.run(training, feed_dict={x: x_train, y_: y_train})
    if i % 10 == 0:
        print('Epoch: ' + str(i) + '  Accuracy = ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) + '  Loss = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})))

# try out an example
sample = 82 # sample number (choose from 0 to TRAIN_SIZE)
answer = sess.run(y, feed_dict={x: x_train})
display_digit(sample) # display image of sample
print(answer[sample]) # print out probabilities of each number
print('Prediction:', answer[sample].argmax()) # print out number with highest probability
