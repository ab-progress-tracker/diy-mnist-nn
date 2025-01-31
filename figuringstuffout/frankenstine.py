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

# dev_size = int(len(test_img) * 0.3) # takes 30% of the data from the 'x_test' set

# print(len(test_img))

# dev_img = test_img[:dev_size]
# dev_lbl = test_lbl[:dev_size]

# test_img = test_img[dev_size:]
# test_lbl = test_lbl[dev_size:]

# print(f"Dev set size: {len(dev_img)}, {len(dev_lbl)}")
# print(f"Actual test set size: {len(test_img)}, {len(test_lbl)}")

# Raw logits (before applying softmax)
logits = np.array([3.0, 2.0, 1.0, 0.1])

# Find the index of the largest logit (class with highest raw score)
predicted_class = np.argmax(logits)

print(f"Predicted class (without softmax): {predicted_class}")

e = np.exp(1)


a_o = (e**logits)/np.sum(e**logits) 
a_alt_o = (logits)/np.sum(logits)
print(a_o, "softmax")
print(np.sum(a_o))
print(a_alt_o, "no softmax")
print(np.sum(a_alt_o))

"""we can see very clearly that softmax exaggerates the confidence for larger values"""


# rnum=random.randint(0,9999)

# print(data['x_train'].shape) #i just wanted to clarify the format it was in bc plt was being pissy
# print(train_img[rnum]) 
# print(train_lbl[rnum])



# image = train_img[rnum]  
# label = train_lbl[rnum] 

# image = image.reshape(28, 28) #figured out why plt was being pissy --- the images are 1D array of 784 pixels

# plt.imshow(image,cmap='binary')
# plt.title(f"Label: {label}")  
# plt.show()

