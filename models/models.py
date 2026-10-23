import numpy as np
from layers.layers import Conv2D, BatchNorm2D, MaxPool2D, Flatten, Dense, ReLU, Dropout, SoftmaxCrossEntropy
from utils.mnist_utils import Mnist

# main classes      
class Model:
  def __init__(self, layers):
    self.layers = layers
    self.training = True
    self.history = {
      "loss": [],
      "acc": [],
      "val_loss": [],
      "val_acc": [],
      "layers": {},
    }

  def train(self):
    self.training = True

  def eval(self):
      self.training = False

  def forward(self, X, Y=None):
    out = X
    for layer in self.layers:
      out = layer.forward(out, track=self.training)
    logits = out

    if Y is None:
      return logits

    # your loss function
    loss = self.loss_fn(logits, Y)
    return loss, logits

  def backward(self):
    grad = self.loss_grad  # whatever your loss gradient is
    for layer in reversed(self.layers):
      grad = layer.backward(grad, track=self.training)

  def get_layers(self):
   return self.layers
  
  def save(self, path):
    #save all parameters to a .npz file
    params_to_save = {}
    idx = 0

    for layer in self.layers:
      params = layer.get_params()
      for p in params:
        params_to_save[f'param_{idx}'] = p
        idx += 1
    
    np.savez(path, **params_to_save)
    print(f'saved model weights to {path}')
  
  def load(self, path):
    #load all parameters from a .npz file
    data = np.load(path)
    keys = sorted(data.files, key=lambda x: int(x.split("_")[1]))

    idx = 0
    for layer in self.layers:
      num_params = len(layer.get_params())
      if num_params == 0:
        continue
      
      layer_params = []
      for _ in range(num_params):
        layer_params.append(data[f'param_{idx}'].copy())
        idx += 1
      
      layer.set_params(layer_params)
    print(f'loaded model weights from {path}')
  
class Trainer:
  def __init__(self, model, optimizer, l2_lambda=0.0):
    self.model = model
    self.optimizer = optimizer
    self.l2_lambda = l2_lambda
    self.epoch_tracker = 1
    self.batch_tracker = 0
    self.history = {
          "loss": [],
          "acc": [],
          "val_loss": [],
          "val_acc": [],
          "layers": {},
        }

  def train_epoch(self, X, Y, batch_size):
    self.model.train()
    batch_losses = []
    batch_accs = []

    for Xb, Yb in Mnist.get_batches(X, Y, batch_size):
      loss, logits, activations = self.model.forward(Xb, Yb)
      self.model.backward()

      #L2 loss
      l2_loss = 0.0
      for layer in self.model.get_layers():
        for W in layer.get_params():
          l2_loss += np.sum(W * W)
      loss_total = loss + self.l2_lambda * l2_loss

      self.optimizer.step(self.model.get_layers())

      batch_losses.append(loss_total)
      batch_accs.append(Mnist.accuracy_from_logits(logits, Yb))

      #collect layer metrics
      for i, layer in enumerate(self.model.get_layers()):
        name = f'layer_{i}_{layer.__class__.__name__}'
        if name not in self.history["layers"]:
          self.history["layers"][name] = {
            "weight_norm": [],
            "grad_norm": [],
            "sparsity": []
          }
        metrics = layer.get_metrics()
        for k, v in metrics.items():
          self.history["layers"][name][k].append(v)

      self.batch_tracker += 1
      print(f'Epoch: {self.epoch_tracker} | Batch: {self.batch_tracker}\n')

    self.epoch_tracker += 1

    return np.mean(batch_losses), np.mean(batch_accs)

  def validate(self, X, Y):
    self.model.eval()
    loss, logits = self.model.forward(X, Y)
    acc = Mnist.accuracy_from_logits(logits, Y)
    return loss, acc

  def train_validate(self,
                      X_train, Y_train_onehot, 
                      X_val_raw, Y_val_onehot, 
                      epochs, batch_size):
      print('Training...')
      for epoch in range(epochs):
        train_loss, train_acc = self.train_epoch(X_train, Y_train_onehot, batch_size)
        val_loss, val_acc = self.validate(X_val_raw, Y_val_onehot)

        self.history["loss"].append(train_loss)
        self.history["acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        print(f'Epoch {epoch+1} | ' 
              f'loss: {train_loss:.4f} | '
              f'acc: {train_acc:.4f} | '
              f'val_loss: {val_loss:.4f} | '
              f'val_acc: {val_acc:.4f}\n'
              )

  def printout(self, epochs, batches_per_epoch, train_imgs, val_imgs):
      #training stats
      print(f'epochs: {epochs} | '
        f'batches per epoch: {batches_per_epoch} | '
        f'total batches: {batches_per_epoch * epochs} | '
        f'images used in training run: {train_imgs} | '
        f'images used in val set: {val_imgs}')
  
# model classes
class MLP(Model):
  #[784(28x28 input image), n..., 10(0-9 classification)]
  def __init__(self):
    #super().__init__()

    self.type = 'mlp'
    self.input_dim = 784
    self.h1_dim = 64
    self.h2_dim = 32
    self.h3_dim = 16 
    self.output_dim = 10

    self.history = {
        "loss": [],
        "acc": [],
        "val_loss": [],
        "val_acc": [],
        "layers": {}
      }
    
    #3 layer
    '''self.layers = [
      Dense(self.input_dim, self.h1_dim),
      ReLU(),
      Dense(self.h1_dim, self.output_dim)
    ]'''

    #4 layer
    '''self.layers = [
      Dense(self.input_dim, self.h1_dim),
      ReLU(),
      Dense(self.h1_dim, self.h2_dim),
      ReLU(),
      Dense(self.h2_dim, self.output_dim)
    ]'''

    #5 layer 
    self.layers = [
      Dense(self.input_dim, self.h1_dim),
      ReLU(),
      Dropout(p=0.1),
      Dense(self.h1_dim, self.h2_dim),
      ReLU(),
      Dense(self.h2_dim, self.h3_dim),
      ReLU(),
      Dense(self.h3_dim, self.output_dim)
    ]

    self.loss_layer = SoftmaxCrossEntropy()

  def forward(self, x, y_true_onehot=None):
    track = self.training
    activations = [] if track else None
    #forward through the layers
    for layer in self.layers:
      x = layer.forward(x, track=track)
      if track and getattr(layer, "is_visualizable", False):
        activations.append((layer.__class__.__name__, layer.get_activation()))
    logits = x
    loss = None
    if y_true_onehot is not None:
      loss = self.loss_layer.forward(logits, y_true_onehot)

    if track:
      return loss, logits, activations
    return loss, logits
  
  def backward(self, track=False):
    track = self.training
    #start from dL/dlogits
    dout = self.loss_layer.backward()
    #backprop through the layers in reverse
    for layer in reversed(self.layers):
      dout = layer.backward(dout, track=track)
  
  def predict(self, x):
    _, logits = self.forward(x, y_true_onehot=None)
    return np.argmax(logits, axis=1)
  
  def get_layers(self):
    return self.layers

class CNN(Model):
  def __init__(self):
    self.type = 'cnn'
    self.c1out = 16
    self.c2out = 32
    self.densesq = 7 # (c1,c2,densesq) (8,16,5) (16,32,7)

    self.layers = [
      Conv2D(1, self.c1out, 3, stride=1, padding=1), 
      BatchNorm2D(self.c1out),
      ReLU(),
      Dropout(p=0.1, mode="channel"),
      MaxPool2D(2, 2),

      Conv2D(self.c1out, self.c2out, 3, stride=1, padding=1),
      BatchNorm2D(self.c2out),
      ReLU(),
      MaxPool2D(2, 2),

      Flatten(),
      Dense(self.c2out*self.densesq*self.densesq, 64),
      ReLU(),
      Dense(64, 10)
    ]
    self.loss_layer = SoftmaxCrossEntropy()

    self.history = {
        "loss": [],
        "acc": [],
        "val_loss": [],
        "val_acc": [],
        "layers": {}
      }

  def forward(self, x, y_true_onehot=None):
    track = self.training
    activations = [] if track else None
    #forward through the layers
    for layer in self.layers:
      x = layer.forward(x, track=track)
      if track and getattr(layer, "is_visualizable", False):
        activations.append((layer.__class__.__name__, layer.get_activation()))
    logits = x
    loss = None
    if y_true_onehot is not None:
      loss = self.loss_layer.forward(logits, y_true_onehot)

    if track:
      return loss, logits, activations
    return loss, logits
  
  def backward(self, track=False):
    track = self.training
    #start from dL/dlogits
    dout = self.loss_layer.backward()
    #backprop through the layers in reverse
    for layer in reversed(self.layers):
      dout = layer.backward(dout, track=track)

  def predict(self, x):
    _, logits = self.forward(x, y_true_onehot=None)
    return np.argmax(logits, axis=1)
  
  def get_layers(self):
    return self.layers
