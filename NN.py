import numpy as np
import h5py
import matplotlib.pyplot as plt
import random

data = h5py.File('MNISTdata.hdf5', 'r')
train_img = np.float16(data['x_train'][:])
train_lbl = np.float16(data['y_train'][:,0])

test_img = np.float16(data['x_test'][:])
test_lbl = np.float16(data['y_test'][:,0])

data.close()

# dev_img = test_img[:dev_size]
# dev_lbl = test_lbl[:dev_size]

# test_img = test_img[dev_size:]
# test_lbl = test_lbl[dev_size:]

class NeuralNetwork:
        def __init__(self):
            # weights and biases connecting the input layer to the hidden layer
            self.weights_input_hidden = np.random.randn(784, 16) * 0.1
            self.biases_input_hidden = np.zeros(16) # sets all biases in hidden layer to 0... fn.
            # weights/biases for hidden to output
            self.weights_hidden_output = np.random.randn(16, 10) * 0.1
            self.biases_hidden_output = np.zeros(10) # sets all biases in output layer to 0... fn.

        def hot_encode(self, labels, num_classes)
            self.numclasses = 10
            hot_encoded = np.eye(num_classes)[labels]

            return hot_encoded

        def forward(self, x): # x is some input pixel
            z_h = np.dot(x, self.weights_input_hidden) + self.biases_input_hidden # np.dot --- matrix multiplication
            # activation function for hidden layer ---- ReLU
            a_h = np.maximum(0, z_h) # it can't just be max() bc it's a vector. oops

            z_o = np.dot(a_h, self.weights_hidden_output) + self.biases_hidden_output
            a_o = np.exp(z_o)/np.sum(np.exp(z_o)) 

            return a_o

        def cost(self, y_pred, y_true)
            cost = np.mean((y_pred-y_true)**2)
            return cost 