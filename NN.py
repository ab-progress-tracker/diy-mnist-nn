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
            a_o = np.exp(z_o)/np.sum(np.exp(z_o), axis = 0) 
            return a_o

        def loss(self, y_pred, y_true):
            cost = np.sum((y_pred-y_true)**2, axis = 0)
            return cost

        def softmax_derivative(self, a_o):
            n = len(a_o)
            s = a_o
            j_m = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    if i == j:
                        # diagonals
                        self.j_m[i, j] = s[i]*(1-s[i])
                    else:
                        # off-diagonal
                        self.j_m[i, j] = s[i]*(-s[j])

        def derivatives(self): # also other vars but i ain't putting em in yet because i'm Like that
            d_relu = (z_h > 0) * 1

            dc_w2 = np.dot(a_h.T, np.dot(j_m, 2*(a_o-y_true)))# derivative of cost function wrt weight 2
            # this basically does: a_h * j_m * 2(a_o-y_true)---checks out if you wanna do the calc. i transposed a_h because it's in the wrong shape otherwise.
            dc_b2 = np.sum(np.dot(j_m, 2*(a_o-y_true)), axis = 0) # derivative of cost function with respect to bias 2

            dc_a_h = np.dot(w2, np.dot(j_m, 2*(a_o-y_true))) # dc wrt activation from hidden
            #w2*a_h*j_m*2(a_o-y_true) 

            # derivative of cost function with respect to weight 1
            dc_w1 = np.dot(x.T, d_relu*dc_a_h)

            # derivative of cost function with respect to bias 1
            dc_b1 = np.sum(d_relu*dc_a_h)
        
        def update_vars(self)
            self.weights_hidden_output = -0.1*dc_w2
            self.biases_hidden_output = -0.1*dc_b2
            self.weights_input_hidden = -0.1*dc_w1
            self.biases_input_hidden = -0.1*dc_b1

        def accuraccy(self, a_o, y_true)
            predictions = np.argmax(a_o)
            accuraccy = np.sum(predictions == y_true) / y_true.size

nn = NeuralNetwork()

# z=nn.forward(10)
# print(z)

# somenumberidk = 3

# for epoch in range(somenumberidk):
#     sumloss = 0
#     for i in range(len(train_img)):
        
#         x = train_img[i]
#         y = int(train_lbl[i])

#         y_pred = nn.forward(x)
#         y_true = nn.hot_encode(y)

#         loss = nn.loss(y_pred, y_true)
#         sumloss += loss

#         # add backprop stuff

#     print(f"Epoch {epoch+1}/{somenumberidk}, cost: {sumloss/len(train_img)}")

