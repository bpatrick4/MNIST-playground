import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
      matrix[t, p] += 1
    return matrix

class DashboardTrain:
  def __init__(self, model):
    self.model = model

  def plot_training_curves(self):
    hist = self.model.history

    plt.figure(figsize=(12,5))

    #loss
    plt.subplot(2,1,1)
    plt.plot(hist["loss"], label='train loss')
    plt.plot(hist["val_loss1"], label='val loss1')
    plt.plot(hist["val_loss2"], label='val loss2')
    plt.title("Loss")
    plt.legend()

    #accuracy
    plt.subplot(2,1,2)
    plt.plot(hist["acc"], label='train acc')
    plt.plot(hist["val_acc1"], label='val acc1')
    plt.plot(hist["val_acc2"], label='val acc2')
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

  def plot_layer_Metrics_cln(self):
    layer_hist = self.model.history['layers']

    for layer_name, metrics in layer_hist.items():

      #determine which metrics have data
      available = {
        name: values
        for name, values in metrics.items()
        if len(values) > 0
      }

      if len(available) == 0:
        continue #nothing to plot

      num_plots = len(available)

      plt.figure(figsize=(5 * num_plots, 4))
      plt.suptitle(layer_name)

      for idx, (metric_name, values) in enumerate(available.items(), start=1):
        plt.subplot(1, num_plots, idx)
        plt.plot(values)
        plt.title(metric_name.replace('-', ' ').title())
      
      plt.tight_layout()
      plt.show()

  def plot_grad_norm_overlays(self):
     layer_hist = self.model.history['layers']

     plt.figure(figsize=(10, 6))

     for layer_name, metrics in layer_hist.items():
        if 'grad_norm' in metrics and len(metrics['grad_norm']) > 0:
           plt.plot(metrics['grad_norm'], label=layer_name)
          
     plt.yscale('log')
     plt.xlabel('Batch')
     plt.ylabel('Gradient Norm (log scale)')
     plt.title('Per-Layer Gradient Norms')
     plt.grid(True)
     plt.legend()
     plt.tight_layout()
     plt.show()
  
  def plot_sparsity_overlays(self):
     layer_hist = self.model.history['layers']

     plt.figure(figsize=(10, 6))

     for layer_name, metrics in layer_hist.items():
        if 'sparsity' in metrics and len(metrics['sparsity']) > 0:
           plt.plot(metrics['sparsity'], label=layer_name)
          
     plt.xlabel('Batch')
     plt.ylabel('Activation Sparsity')
     plt.title('Per-Layer Activation Sparsity')
     plt.grid(True)
     plt.legend()
     plt.tight_layout()
     plt.show()

  def plot_weight_norm_overlays(self):
     layer_hist = self.model.history['layers']

     plt.figure(figsize=(10, 6))

     for layer_name, metrics in layer_hist.items():
        if 'weight_norm' in metrics and len(metrics['weight_norm']) > 0:
           plt.plot(metrics['weight_norm'], label=layer_name)
          
     plt.xlabel('Batch')
     plt.ylabel('Weight Norm')
     plt.title('Per-Layer Weight Norms')
     plt.grid(True)
     plt.legend()
     plt.tight_layout()
     plt.show()

  def plot_all_overlays(self):
     self.plot_grad_norm_overlays()
     self.plot_sparsity_overlays()
     self.plot_weight_norm_overlays()
       
class DashboardTest:
  def __init__(self, tester):
    self.tester = tester
    self.model = tester.model
  
  def plot_test_confusion(self, X, y_true):
    #confusion matrix
    preds = self.model.predict(X)
    cm = confusion_matrix(y_true, preds)

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1) 
    plt.imshow(cm, cmap='viridis')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    
    #per class accuracy
    accs = [] 
    
    for cls in range(10): 
      mask = (y_true == cls) 
      acc = np.mean(preds[mask] == cls) 
      accs.append(acc)

    plt.subplot(1,2,2) 
    plt.bar(range(10), accs) 
    plt.title("Per-Class Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.grid()
    plt.show()

  def plot_test_feature_maps(self, x_single):
    #forward pass with tracking
    _, _, activations = self.model.forward(x_single, track=True)

    for name, fmap in activations:
      #remove batch dimension if present
      if fmap.ndim == 4:
        fmap = fmap[0]
        num = fmap.shape[0]

        cols = 8
        rows = (num + cols -1) // cols

        plt.figure(figsize=(12, 2 * rows))
        plt.suptitle(f'{name} feature maps')

        for i in range(num):
          plt.subplot(rows, cols, i + 1)
          plt.imshow(fmap[i], cmap='viridis')
          plt.axis('off')

        plt.tight_layout()
        plt.show()

      elif fmap.ndim == 2:
        #dense layer: (batch, units)
        vec = fmap[0] #take first sample
        units = vec.shape[0]

        plt.figure(figsize=(12, 4))
        plt.suptitle(f'{name} activations')
        plt.bar(np.arange(units), vec)
        plt.xlabel('neuron index')
        plt.ylabel('activation')
        plt.tight_layout()
        plt.show()

      elif fmap.ndim == 1:
        #already a vector
        units = vec.shape[0]

        plt.figure(figsize=(12, 4))
        plt.suptitle(f'{name} activations')
        plt.bar(np.arange(units), fmap)
        plt.xlabel('neuron index')
        plt.ylabel('activation')
        plt.tight_layout()
        plt.show()

      else:
        #fallback: show as heatmap
        arr = fmap.squeeze()
        plt.figure(figsize=(10, 3))
        plt.title(f'{name} activation (heatmap fallback)')
        plt.imshow(arr[np.newaxis, :], aspect='auto', cmap='viridis')
        plt.tight_layout()
        plt.show()
