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

dev_percentage = 0.3

dev_size = int(len(test_img) * dev_percentage)

print(len(test_img))

dev_img = test_img[:dev_size]
dev_lbl = test_lbl[:dev_size]

test_img = test_img[dev_size:]
test_lbl = test_lbl[dev_size:]

print(f"Dev set size: {len(dev_img)}, {len(dev_lbl)}")
print(f"Actual test set size: {len(test_img)}, {len(test_lbl)}")


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

