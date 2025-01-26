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

        def hot_encode(self, y, numclasses=10):
            hot_encoded = np.eye(numclasses)[y]
            return hot_encoded


        def forward(self, x): # x is some input pixel
            z_h = np.dot(x, self.weights_input_hidden) + self.biases_input_hidden # np.dot --- matrix multiplication
            # activation function for hidden layer ---- ReLU
            a_h = np.maximum(0, z_h) # it can't just be max() bc it's a vector. oops

            z_o = np.dot(a_h, self.weights_hidden_output) + self.biases_hidden_output
            a_o = np.exp(z_o)/np.sum(np.exp(z_o)) 

            return a_o

        def cost(self, y_pred, y_true):
            cost = np.sum((y_pred-y_true)**2)
            return cost 


nn = NeuralNetwork()

somenumberidk = 3

for epoch in range(somenumberidk):
    avg_loss=0
    for i in range(len(train_img)):
        
        x = train_img[i]
        y = int(train_lbl[i])

        y_pred = nn.forward(x)
        y_true = nn.hot_encode(y)

        loss = nn.cost(y_pred, y_true)
        avg_loss = avg_loss + loss

        # add backprop stuff

    print(f"Epoch {epoch+1}/{somenumberidk}, loss: {avg_loss/len(train_img)}")

