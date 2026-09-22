import numpy as np
from layers.layers import Conv2D, BatchNorm2D, MaxPool2D, Flatten, Dense, ReLU, SoftmaxCrossEntropy
from mnist_data.mnist_utils import Mnist

# main classes      
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
  
class Train:
    def printout(self, epochs, batches_per_epoch, train_imgs, val_imgs):
      #training stats
      print(f'epochs: {epochs} | '
      f'batches per epoch: {batches_per_epoch} | '
      f'total batches: {batches_per_epoch * epochs} | '
      f'images used in training run: {train_imgs} | '
      f'images used in val set: {val_imgs}')

    def train_epoch(self, X, Y, optimizer, lr, batch_size, l2_lambda=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
      batch_losses = []
      batch_accs = []
      batch_tracker = 0
      t = 0
        
      for Xb, Yb in Mnist.get_batches(X, Y, batch_size):
        loss, logits, activations = self.forward(Xb, Yb, track=True)
        self.backward(track=True)

        #L2 loss
        l2_loss = 0.0
        for layer in self.get_layers():
          for W in layer.get_params():
            l2_loss += np.sum(W * W)
        loss_total = loss + l2_lambda * l2_loss

        if optimizer == 'sgd':
          #SGD update
          for layer in self.get_layers():
            for p, g in zip(layer.get_params(), layer.get_grads()):
              p -= lr * g
        elif optimizer == 'adam':
          #Adam update
          t += 1
          for layer in self.get_layers():
            params = layer.get_params()
            grads = layer.get_grads()

            if not hasattr(layer, 'm'):
              layer.init_optimizer_state()
            
            for i, (p, g) in enumerate(zip(params, grads)):
              #update biased first moment estimate
              layer.m[i] = beta1 * layer.m[i] + (1 - beta1) * g

              #update biased second raw moment estimate
              layer.v[i] = beta2 * layer.v[i] + (1 - beta2) * (g * g)

              #compute bias-corrected moments
              m_hat = layer.m[i] / (1 - beta1 ** t)
              v_hat = layer.v[i] / (1 - beta2 ** t)

              #update parameters
              p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        batch_losses.append(loss_total)
        batch_accs.append(Mnist.accuracy_from_logits(logits, Yb))

        #collect layer metrics
        for i, layer in enumerate(self.get_layers()):
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
    
        batch_tracker += 1
        print(f'Batch: {batch_tracker}\n')
      
      return np.mean(batch_losses), np.mean(batch_accs)
    
    def validate(self, X, Y):
      loss, logits = self.forward(X, Y, track=False)
      acc = Mnist.accuracy_from_logits(logits, Y)
      #print('test set accuracy:', acc)
      return loss, acc

    def train_validate(self,
                      X_train, Y_train_onehot, X_val_raw, Y_val_onehot, 
                      epochs, optimizer, lr, batch_size, l2_lambda=0.0, 
                      beta1=0.9, beta2=0.999, eps=1e-8):
      print('Training...')
      for epoch in range(epochs):
        train_loss, train_acc = self.train_epoch(X_train, Y_train_onehot, optimizer, lr, batch_size, l2_lambda, beta1, beta2, eps)
        val_loss, val_acc = self.validate(X_val_raw, Y_val_onehot)

        self.history["loss"].append(train_loss)
        self.history["acc"].append(train_acc)
        self.history["val_loss1"].append(val_loss)
        self.history["val_acc1"].append(val_acc)

        print(f'Epoch {epoch+1} | ' 
              f'loss: {train_loss:.4f} | '
              f'acc: {train_acc:.4f} | '
              f'val_loss: {val_loss:.4f} | '
              f'val_acc: {val_acc:.4f}\n'
              )

    def train_validate_2(self, epochs, trainer, X_train, Y_train_onehot, X_val_raw1, Y_val_onehot1, X_val_raw2, Y_val_onehot2):
      print('Training...')
      for epoch in range(epochs):
        train_loss, train_acc = trainer.train_epoch(X_train, Y_train_onehot)
        val_loss1, val_acc1 = self.validate(X_val_raw1, Y_val_onehot1)
        val_loss2, val_acc2 = self.validate(X_val_raw2, Y_val_onehot2)


        trainer.history["loss"].append(train_loss)
        trainer.history["acc"].append(train_acc)
        trainer.history["val_loss1"].append(val_loss1)
        trainer.history["val_acc1"].append(val_acc1)
        trainer.history["val_loss2"].append(val_loss2)
        trainer.history["val_acc2"].append(val_acc2)

        print(f'Epoch {epoch+1} | ' 
              f'loss: {train_loss:.4f} | '
              f'acc: {train_acc:.4f} | '
              f'val_loss1: {val_loss1:.4f} | '
              f'val_acc1: {val_acc1:.4f} | '
              f'val_loss2: {val_loss2:.4f} | '
              f'val_acc2: {val_acc2:.4f}'
              )

class Tester_mlp:
  def __init__(self, model, test_sets):
    self.model = model
    self.type = self.model.type
    
    self.X_test_raw = test_sets[0]
    self.Y_test_raw = test_sets[1]
    self.x_single_raw = test_sets[2]
    self.Y_test_onehot = test_sets[3]
   
  def run_test(self, X, y_true):
    #validate for accuracy
    return self.model.validate(X, y_true)[1]

class Tester_cnn:
  def __init__(self, model, test_sets):
    self.model = model
    self.type = self.model.type
    
    self.X_test_small_cnn = test_sets[0]
    self.Y_test_small_raw = test_sets[1]
    self.x_single_cnn = test_sets[2]
    self.Y_test_small_onehot = test_sets[3]
   
  def run_test(self, X, y_true):
    #validate for accuracy
    return self.model.validate(X, y_true)[1]

# model classes
class MLP(Model, Train):
  #[784(28x28 input image), n..., 10(0-9 classification)]
  def __init__(self):
    self.type = 'mlp'
    self.input_dim = 784
    self.h1_dim = 64
    self.h2_dim = 32
    self.h3_dim = 16 
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
      BatchNorm2D(self.c1out),
      ReLU(),
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
