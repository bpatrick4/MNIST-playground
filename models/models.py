import numpy as np
from layers.layers import Conv2D, MaxPool2D, Flatten, Dense, ReLU, SoftmaxCrossEntropy # BatchNorm2D
from trainer.trainer import Train

class Model:
  def forward(self, x, y=None, track=False):
    raise NotImplementedError
  
  def backward(self, track=False):
    raise NotImplementedError
  
  def predict(self, x):
    raise NotImplementedError
  
  def get_layers(self):
    raise NotImplementedError
  
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
  
class MLP(Model, Train):
  #[784(28x28 input image), n..., 10(0-9 classification)]
  def __init__(self):
    self.type = 'mlp'
    self.input_dim = 784
    self.h1_dim = 32
    self.h2_dim = 64
    self.h3_dim = 10 
    self.output_dim = 10
    self.history = {
        "loss": [],
        "acc": [],
        "val_loss1": [],
        "val_acc1": [],
        "val_loss2": [],
        "val_acc2": [],
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
      Dense(self.h1_dim, self.h2_dim),
      ReLU(),
      Dense(self.h2_dim, self.h3_dim),
      ReLU(),
      Dense(self.h3_dim, self.output_dim)
    ]

    self.loss_layer = SoftmaxCrossEntropy()

  def forward(self, x, y_true_onehot=None, track=False):
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
    #start from dL/dlogits
    dout = self.loss_layer.backward()
    #backprop through the layers in reverse
    for layer in reversed(self.layers):
      dout = layer.backward(dout, track=track)
  
  def predict(self, x):
    _, logits = self.forward(x, y_true_onehot=None, track=False)
    return np.argmax(logits, axis=1)
  
  def get_layers(self):
    return self.layers

class CNN(Model, Train):
  def __init__(self):
    self.type = 'cnn'
    self.c1out = 16
    self.c2out = 32
    self.densesq = 7 # (c1,c2,densesq) (8,16,5) (16,32,7)

    self.layers = [
      Conv2D(1, self.c1out, 3, stride=1, padding=1), 
      #BatchNorm2D(self.c1out),
      ReLU(),
      MaxPool2D(2, 2),

      Conv2D(self.c1out, self.c2out, 3, stride=1, padding=1),
      #BatchNorm2D(self.c2out),
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
        "val_loss1": [],
        "val_acc1": [],
        "val_loss2": [],
        "val_acc2": [],
        "layers": {}
      }

  def forward(self, x, y_true_onehot=None, track=False):
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
    #start from dL/dlogits
    dout = self.loss_layer.backward()
    #backprop through the layers in reverse
    for layer in reversed(self.layers):
      dout = layer.backward(dout, track=track)

  def predict(self, x):
    _, logits = self.forward(x, y_true_onehot=None, track=False)
    return np.argmax(logits, axis=1)
  
  def get_layers(self):
    return self.layers
