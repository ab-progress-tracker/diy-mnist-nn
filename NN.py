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

dev_size = int(len(test_img) * 0.3)
dev_img = test_img[:dev_size]
dev_lbl = test_lbl[:dev_size]

test_img = test_img[dev_size:]
test_lbl = test_lbl[dev_size:]

class NeuralNetwork:
        def __init__(self):
            # weights and biases connecting the input layer to the hidden layer
            self.weights_input_hidden = np.random.randn(784, 32) * 0.01
            self.biases_input_hidden = np.zeros(32) # sets all biases in hidden layer to 0... fn.
            # weights/biases for hidden to output
            self.weights_hidden_output = np.random.randn(32, 10) * 0.01
            self.biases_hidden_output = np.zeros(10) # sets all biases in output layer to 0... fn.

        def hot_encode(self, y, numclasses=10):
            hot_encoded = np.eye(numclasses)[y]
            return hot_encoded

        def forward(self, x): # x is some input pixel
            z_h = np.dot(x, self.weights_input_hidden) + self.biases_input_hidden # np.dot --- matrix multiplication
            # activation function for hidden layer ---- ReLU
            a_h = np.maximum(0, z_h) # it can't just be max() bc it's a vector. oops
    
            z_o = np.dot(a_h, self.weights_hidden_output) + self.biases_hidden_output
            z_s = z_o - np.max(z_o) # to prevent overflow --- z_s = z stable

            a_o = np.exp(z_s)/np.sum(np.exp(z_s), axis = 0) 
            return z_h, a_h, z_o, a_o

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
                        j_m[i, j] = s[i]*(1-s[i])
                    else:
                        # off-diagonal
                        j_m[i, j] = s[i]*(-s[j])
            return j_m

        def derivatives(self, z_h, a_h, z_o, a_o, x, y_true, j_m): # also other vars but i ain't putting em in yet because i'm Like that
            w2 = self.weights_hidden_output

            d_relu = (z_h > 0) * 1

            dc_w2 = np.outer(a_h, 2 * (a_o - y_true))# derivative of cost function wrt weight 2
            # this basically does: a_h * j_m * 2(a_o-y_true)---checks out if you wanna do the calc. i transposed a_h because it's in the wrong shape otherwise.
            dc_b2 = np.sum(np.dot(j_m, 2*(a_o-y_true)), axis = 0) # derivative of cost function with respect to bias 2

            dc_a_h = np.dot(w2, np.dot(j_m, 2*(a_o-y_true))) # dc wrt activation from hidden
            #w2*a_h*j_m*2(a_o-y_true) 

            # derivative of cost function with respect to weight 1
            dc_w1 = np.outer(x, d_relu*dc_a_h)

            # derivative of cost function with respect to bias 1
            dc_b1 = np.sum(d_relu*dc_a_h)

            return dc_w2, dc_b2, dc_w1, dc_b1
        
        def update_vars(self, dc_w2, dc_b2, dc_w1, dc_b1, a):
            self.weights_hidden_output += -(a*dc_w2)
            self.biases_hidden_output += -(a*dc_b2)
            self.weights_input_hidden += -(a*dc_w1)
            self.biases_input_hidden += -(a*dc_b1)

        def accuraccy(self, a_o, y_true):
            predictions = np.argmax(a_o, axis = 0)
            accuraccy = np.sum(predictions == y_true) / y_true.size
            return accuraccy

nn = NeuralNetwork()

# z=nn.forward(10)
# print(z)

ex_plot = list()
ey_plot = list()
somenumberidk = 5
sumloss = 0
learning_rate = 0.01

for epoch in range(somenumberidk):
    sumloss = 0
    epoch_accuracy = 0  
    learning_rate *= 0.9
    for i in range(len(train_img)):
        
        x = train_img[i]
        y = int(train_lbl[i])

        z_h, a_h, z_o, a_o = nn.forward(x)
        j_m = nn.softmax_derivative(a_o)

        y_pred = a_o
        y_true = nn.hot_encode(y)

        loss = nn.loss(y_pred, y_true)
        sumloss += loss

        dc_w2, dc_b2, dc_w1, dc_b1 = nn.derivatives(z_h, a_h, z_o, a_o, x, y_true, j_m)

        nn.update_vars(dc_w2, dc_b2, dc_w1, dc_b1, learning_rate)

        if np.argmax(y_pred) == np.argmax(y_true):  
            epoch_accuracy += 1
    epoch_accuracy = epoch_accuracy/len(train_img)

    ex_plot.append(epoch+1)
    ey_plot.append(epoch_accuracy)

    print(f"Epoch {epoch+1}/{somenumberidk}, accuracy: {epoch_accuracy}.") # loss: {sumloss/len(train_img)},

print(ex_plot)
print(ey_plot)