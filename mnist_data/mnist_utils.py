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
