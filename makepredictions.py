import numpy as np
import h5py
import random
import matplotlib.pyplot as plt

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

# Load the .npz file
params = np.load('model_params.npz')

# Access the variables (weights and biases)
w1 = params['weights_input_hidden']
b1 = params['biases_input_hidden']
w2 = params['weights_hidden_output']
b2 = params['biases_hidden_output']

def forward(x): # x is some input pixel
    z_h = np.dot(x, w1) + b1 # np.dot --- matrix multiplication
    # activation function for hidden layer ---- ReLU
    a_h = np.maximum(0, z_h) # it can't just be max() bc it's a vector. oops

    z_o = np.dot(a_h, w2) + b2
    z_s = z_o - np.max(z_o) # to prevent overflow --- z_s = z stable

    a_o = np.exp(z_s)/np.sum(np.exp(z_s), axis = 0) 
    return a_o

rnum = random.randint(0, len(dev_img) - 1)
img = dev_img[rnum]
lbl = dev_lbl[rnum]

softpreds = forward(img)

prediction = np.argmax(softpreds)

img = img.reshape(28, 28) 
plt.imshow(img,cmap='binary')
plt.title(f"Predicted: {prediction}\nTrue label: {lbl}")  
plt.show()
