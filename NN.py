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

dev_img = test_img[:dev_size]
dev_lbl = test_lbl[:dev_size]

test_img = test_img[dev_size:]
test_lbl = test_lbl[dev_size:]

class NeuralNetwork:
        def __init__(self):
            # weights and biases connecting the input layer to the hidden layer
            self.weights_input_hidden = np.random.randn(784, 16) * 0.1
            self.biases_input_hidden = np.zeros(16) # sets all biases in hidden layer to 0... fn.
            # weights/biases for hidden to output
            self.weights_hidden_output = np.random.randn(16, 10) * 0.1
            self.biases_hidden_output = np.zeros(10) # sets all biases in output layer to 0... fn.
        
        def forward(self, x): # x is some input pixel
            z_h = np.dot(x, self.weights_input_hidden) + self.biases_input_hidden # np.dot --- matrix multiplication
            # activation function for hidden layer ---- ReLU
            a_h = max(0, z_h)

            z_o = np.dot(a_h, self.weights_hidden_output) + self.biases_hidden_output
            e = np.exp(1) # euler's number
            a_o = (e**z_o)/np.sum(e**z_o) # this is softmax. 

            return a_o 



            
