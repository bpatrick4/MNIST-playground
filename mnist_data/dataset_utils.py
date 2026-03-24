from mnist_data.mnist_utils import Mnist
import random

#set dataset inputs
num_of_train_imgs = 48000 #2 ** 11
num_of_val_imgs = 12000 #2 ** 10
num_of_test_imgs = 100
n =  random.randint(0,9999)
n = 4443

#outputs
train_ratio = 1 - (num_of_train_imgs / 60000)
val_ratio = 1 - (num_of_val_imgs / 60000)

#load mnist dataset
path = 'mnist_data' #download mnist dataset to this folder

#raw datasets
X_train_raw, Y_train_raw = Mnist.load_mnist(path, kind='train')
X_test_raw, Y_test_raw = Mnist.load_mnist(path, kind='t10k')

#curated datasets
X_train, Y_train, X_val_raw, Y_val_raw = Mnist.get_val_set(X_train_raw, Y_train_raw, train_ratio) #split train set into train/val
X_val_raw1, Y_val_raw1, X_val_raw2, Y_val_raw2 = Mnist.get_val_set(X_val_raw, Y_val_raw, val_ratio) #split val set for quicker CNN training

X_test_small_raw, Y_test_small_raw = X_test_raw[0:num_of_test_imgs], Y_test_raw[0:num_of_test_imgs] #split test set for quicker CNN testing
x_single_raw = X_test_raw[n:n+1] #pick an image for visualization

#one-hot encode labels
Y_train_onehot = Mnist.one_hot_encode(Y_train)
Y_val_onehot = Mnist.one_hot_encode(Y_val_raw) #Y_val_raw1 + Y_val_raw2 = Y_val_raw
Y_val_onehot1 = Mnist.one_hot_encode(Y_val_raw1) 
Y_val_onehot2 = Mnist.one_hot_encode(Y_val_raw2) 

Y_test_onehot = Mnist.one_hot_encode(Y_test_raw)
Y_test_small_onehot = Mnist.one_hot_encode(Y_test_small_raw)

#reshape MNIST images for CNN 
X_train_cnn = X_train.reshape(-1, 1, 28, 28) 
X_val_cnn1 = X_val_raw1.reshape(-1, 1, 28, 28)
X_val_cnn2 = X_val_raw2.reshape(-1, 1, 28, 28)

X_test_cnn = X_test_raw.reshape(-1, 1, 28, 28)
X_test_small_cnn = X_test_small_raw.reshape(-1, 1, 28, 28)
x_single_cnn = x_single_raw.reshape(-1, 1, 28, 28)

train_sets = [X_train, Y_train, 
         X_val_raw, Y_val_raw, 
         X_val_raw1, Y_val_raw1, X_val_raw2, Y_val_raw2, 
         Y_train_onehot, Y_val_onehot, Y_val_onehot1, Y_val_onehot2, 
         X_train_cnn, X_val_cnn1, X_val_cnn2]

test_sets = [X_test_raw, Y_test_raw, 
        X_test_small_raw, Y_test_small_raw, x_single_raw, 
        Y_test_onehot, Y_test_small_onehot, 
        X_test_cnn, X_test_small_cnn, x_single_cnn]

dataset_inputs = [num_of_train_imgs, num_of_val_imgs, num_of_test_imgs, n]
