import numpy as np
import gzip
import os

class Mnist:
  # load MNIST dataset from raw files
  def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
      labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
      images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images / 255.0, labels

  # one hot encoding for labels
  def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot
  
  # validation set
  def get_val_set(X_train_raw, Y_train_raw, val_ratio):
    m = X_train_raw.shape[0]
    indices = np.random.permutation(m)
    split = int(m * val_ratio)

    val_idx = indices[:split]
    train_idx = indices[split:]

    X_val_raw = X_train_raw[val_idx]
    Y_val_raw = Y_train_raw[val_idx]

    X_train = X_train_raw[train_idx]
    Y_train = Y_train_raw[train_idx]

    return X_train, Y_train, X_val_raw, Y_val_raw

  # validation set batching
  def get_val_batches(X, Y, batch_size):
    m = X.shape[0]
    for i in range(0, m, batch_size):
      X_batch = X[i:i+batch_size]
      Y_batch = Y[i:i+batch_size]
      yield X_batch, Y_batch

  # batching
  def get_batches(X, Y, batch_size):
    m = X.shape[0]
    indices = np.random.permutation(m)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    for i in range(0, m, batch_size):
      X_batch = X_shuffled[i:i+batch_size]
      Y_batch = Y_shuffled[i:i+batch_size]
      yield X_batch, Y_batch

  # testing/validation
  def accuracy_from_logits(logits, y_onehot):
    preds = np.argmax(logits, axis=1)
    true = np.argmax(y_onehot, axis=1)
    return np.mean(preds == true)

  # find numbers for testing
  def num_finder(num, samples):
      path = 'mnist_data' #download mnist dataset to this folder
      _, Y_test_raw = Mnist.load_mnist(path, kind='t10k')
      idxs = []
      for i in range(0,len(Y_test_raw)):
        if Y_test_raw[i]  == num:
          idxs.append(i)
          af = idxs[0:samples]
      print(af)

# 10 of each number
#0: [3, 10, 13, 25, 28, 55, 69, 71, 101, 126]
#1: [2, 5, 14, 29, 31, 37, 39, 40, 46, 57]
#2: [1, 35, 38, 43, 47, 72, 77, 82, 106, 119]
#3: [18, 30, 32, 44, 51, 63, 68, 76, 87, 90]
#4: [4, 6, 19, 24, 27, 33, 42, 48, 49, 56]
#5: [8, 15, 23, 45, 52, 53, 59, 102, 120, 127]
#6: [11, 21, 22, 50, 54, 66, 81, 88, 91, 98]
#7: [0, 17, 26, 34, 36, 41, 60, 64, 70, 75]
#8: [61, 84, 110, 128, 134, 146, 177, 179, 181, 184]
#9: [7, 9, 12, 16, 20, 58, 62, 73, 78, 92]

# tricky images
#0: 7216
#1: 2221
#2: 
#3: 4443
#4: 3996
#5: 8, 9970
#6: 
#7: 
#8: 
#9: 833