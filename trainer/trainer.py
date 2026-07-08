import numpy as np
from mnist_data.mnist_utils import Mnist

class Trainer:
  def __init__(self, model, optimizer, lr, batch_size, l2_lambda=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
    self.model = model
    self.optimizer = optimizer
    self.lr = lr
    self.batch_size = batch_size
    self.l2_lambda = l2_lambda
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.t = 0

    self.history = {
      "loss": [],
      "acc": [],
      "val_loss1": [],
      "val_acc1": [],
      "val_loss2": [],
      "val_acc2": [],
      "layers": {}
    }

  def printout(self, epochs, batches_per_epoch, train_imgs, val_imgs):
    #training stats
    print(f'epochs: {epochs} | '
      f'batches per epoch: {batches_per_epoch} | '
      f'total batches: {batches_per_epoch * epochs} | '
      f'images used in training run: {train_imgs} | '
      f'images used in val set: {val_imgs}')
    
  def train_epoch(self, X, Y):
    batch_losses = []
    batch_accs = []
    batch_tracker = 0
      
    for Xb, Yb in Mnist.get_batches(X, Y, self.batch_size):
      loss, logits, activations = self.model.forward(Xb, Yb, track=True)
      self.model.backward(track=True)

      #L2 loss
      l2_loss = 0.0
      for layer in self.model.get_layers():
        for W in layer.get_params():
          l2_loss += np.sum(W * W)
      loss_total = loss + self.l2_lambda * l2_loss

      if self.optimizer == 'sgd':
        #SGD update
        for layer in self.model.get_layers():
          for p, g in zip(layer.get_params(), layer.get_grads()):
            p -= self.lr * g
      elif self.optimizer == 'adam':
        #Adam update
        self.t += 1
        for layer in self.model.get_layers():
          params = layer.get_params()
          grads = layer.get_grads()

          if not hasattr(layer, 'm'):
            layer.init_optimizer_state()
          
          for i, (p, g) in enumerate(zip(params, grads)):
            #update biased first moment estimate
            layer.m[i] = self.beta1 * layer.m[i] + (1 - self.beta1) * g

            #update biased second raw moment estimate
            layer.v[i] = self.beta2 * layer.v[i] + (1 - self.beta2) * (g * g)

            #compute bias-corrected moments
            m_hat = layer.m[i] / (1 - self.beta1 ** self.t)
            v_hat = layer.v[i] / (1 - self.beta2 ** self.t)

            #update parameters
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

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
  
      batch_tracker += 1
      print(f'Batch: {batch_tracker}')
    

    return np.mean(batch_losses), np.mean(batch_accs)

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
      