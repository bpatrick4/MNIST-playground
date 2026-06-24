import numpy as np

class Layer:
  def forward(self, x, track=False):
    raise NotImplementedError
  
  def backward(self, dout, track=False):
    raise NotImplementedError
  
  def set_params(self, params):
    raise NotImplementedError
    
  def get_params(self):
    return []
  
  def get_grads(self):
    return []
  
  def get_metrics(self):
    return {}
  
  def init_optimizer_state(self):
    #called once before first Adam update
    self.m = [np.zeros_like(p) for p in self.get_params()]
    self.v = [np.zeros_like(p) for p in self.get_params()]

#MLP
class Dense(Layer): 
  is_visualizable = True
  
  def __init__(self, in_dim, out_dim):
    #He init
    scale = np.sqrt(2.0 / in_dim)
    self.W = np.random.randn(in_dim, out_dim) * scale
    self.b = np.zeros((1, out_dim))

    self.activation = None

  def forward(self, x, track=False):
    self.x = x #cache for backward
    out = x @ self.W + self.b
    if track:
      self.activation = out.copy()
      self.last_out = out
    else:
      self.activation = None
      
    return out
  
  def backward(self, dout, track=False):
    #dout: gradient of loss w.r.t this layer's output
    self.dW = self.x.T @ dout
    self.db = np.sum(dout, axis=0, keepdims=True)
    dx = dout @ self.W.T
    if track:
      self.last_dout = dout
    return dx
  
  def get_activation(self):
    return self.activation
  
  def set_params(self, params):
    self.W, self.b = params
    
  def get_params(self):
    return [self.W, self.b]
  
  def get_grads(self):
    return [self.dW, self.db]
  
  def get_metrics(self):
    return {
      "weight_norm": np.linalg.norm(self.W),
      "grad_norm": np.linalg.norm(self.dW) if hasattr(self, "dW") else 0.0,
    } 

class ReLU(Layer):  
  def forward(self, x, track=False):
    self.mask = (x > 0)
    if track:
      self.last_x = x
    return x * self.mask
  
  def backward(self, dout, track=False):
    return dout * self.mask
  
  def set_params(self, params):
    pass
    
  def get_params(self):
    return []
  
  def get_grads(self):
    return []
  
  def get_metrics(self):
    if hasattr(self, "mask"):
      sparsity = 1.0 - np.mean(self.mask)
    else:
      sparsity = 0.0
    return {"sparsity": sparsity}

class SoftmaxCrossEntropy(Layer):
  def forward(self, logits, y_true_onehot):
    #logits: (batch, num_classes)
    #y_true_onehot: (batch, num_classes)
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    self.probs = exps / np.sum(exps, axis=1, keepdims=True)
    self.y_true = y_true_onehot

    m = y_true_onehot.shape[0]
    loss = -np.sum(y_true_onehot * np.log(self.probs + 1e-8)) / m
    return loss
  
  def backward(self):
    #gradient of loss w.r.t. to logits
    m = self.y_true.shape[0]
    return (self.probs - self.y_true) / m
  
  def set_params(self, params):
    pass
    
  def get_params(self):
    return []
  
  def get_grads(self):
    return []
  
#CNN
class Conv2D(Layer):
  is_visualizable = True
  
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    # Xavier/He-like init
    scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
    self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
    self.b = np.zeros((out_channels, 1))
    
    self.activation = None

  def forward(self, x, track=False):
    #x shape: (batch, in_channels, H, W)
    self.x = x
    batch, in_channels, H, W = x.shape
    k = self.kernel_size
    S = self.stride
    P = self.padding

    #pad input
    if P > 0:
      x_padded = np.pad(
        x,
        ((0,0), (0,0), (P,P), (P,P)),
        mode='constant'
      )
    else:
      x_padded = x
    H_p, W_p = x_padded.shape[2], x_padded.shape[3]
    
    out_h = (H_p - k) // S + 1
    out_w = (W_p - k) // S + 1
    
    #im2col: unfold all sliding windows
    cols = np.zeros((batch, in_channels * k * k, out_h * out_w))

    for n in range(batch):
      col = []
      for i in range(out_h):
        for j in range(out_w):
          h_start = i * S
          w_start = j * S
          patch = x_padded[n, :, h_start:h_start+k, w_start:w_start+k].reshape(-1)
          col.append(patch)
      cols[n] = np.stack(col, axis=1)

    #reshape filter
    W_col = self.W.reshape(self.out_channels, -1)

    #matrix multiply
    out = np.zeros((batch, self.out_channels, out_h * out_w))
    for n in range(batch):
      out[n] = W_col @ cols[n] + self.b

    #reshape to (batch, self.out_channels, out_h, out_w)
    out = out.reshape(batch, self.out_channels, out_h, out_w)
    
    if track:
      self.activation = out.copy()
      self.last_out = out
    else:
      self.activation = None

    return out
    
  def backward(self, dout, track=False):
    x = self.x
    batch, in_channels, H, W = x.shape
    _, _, out_h, out_w = dout.shape
    k = self.kernel_size
    S = self.stride
    P = self.padding

    #pad input
    if P > 0:
      x_padded = np.pad(
        x,
        ((0,0), (0,0), (P,P), (P,P)),
        mode='constant'
      )
    else:
      x_padded = x

    H_p, W_p = x_padded.shape[2], x_padded.shape[3]

    #reshape dout
    dout_flat = dout.reshape(batch, self.out_channels, -1)

    #gradients
    dW = np.zeros_like(self.W)
    db = np.sum(dout, axis=(0,2,3)).reshape(self.b.shape)
    dx_padded = np.zeros_like(x_padded)

    W_col = self.W.reshape(self.out_channels, -1)

    for n in range(batch):
      #im2col for input
      cols = []
      for i in range(out_h):
        for j in range(out_w):
          h_start = i * S
          w_start = j * S
          patch = x_padded[n, :, h_start:h_start+k, w_start:w_start+k].reshape(-1)
          cols.append(patch)
      cols = np.stack(cols, axis=1)

      #dW
      dW += (dout_flat[n] @ cols.T).reshape(self.W.shape)

      #dx via col2im
      dcols = W_col.T @ dout_flat[n]
      dcols = dcols.reshape(in_channels, k, k, out_h, out_w)

      for i in range(out_h):
        for j in range(out_w):
          h_start = i * S
          w_start = j * S
          dx_padded[n, :, h_start:h_start+k, w_start:w_start+k] += dcols[:, :, :, i, j]

    #remove padding
    if P > 0:
      dx = dx_padded[:, :, P:-P, P:-P]
    else:
      dx = dx_padded
    
    #save gradients
    self.dW = dW
    self.db = db

    if track:
      self.last_dout = dout

    return dx

  def get_activation(self):
    return self.activation
  
  def set_params(self, params):
    self.W, self.b = params

  def get_params(self):
    return [self.W, self.b]
  
  def get_grads(self):
    return [self.dW, self.db]
  
  def get_metrics(self):
    return {
      "weight_norm": np.linalg.norm(self.W),
      "grad_norm": np.linalg.norm(self.dW) if hasattr(self, "dW") else 0.0,
    } 

class BatchNorm2D(Layer): #needs work
  def __init__(self, num_features, eps=1e-5, momentum=0.9):
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum

    #learnable parameters
    self.gamma = np.ones((1, num_features, 1, 1))
    self.beta = np.zeros((1, num_features, 1, 1))

    #running stats
    self.running_mean = np.zeros((1, num_features, 1, 1))
    self.running_var = np.ones((1, num_features, 1, 1))

  def forward(self, x, track=True):
    self.x = x
    if track: #training mode
      mean = np.mean(x, axis=(0,2,3), keepdims=True)
      var = np.var(x, axis=(0,2,3), keepdims=True)

      #update running stats
      self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
      self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

      self.mean = mean
      self.var = var

      x_hat = (x - mean) / np.sqrt(var + self.eps)
    else: #inference mode
      x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

    out = self.gamma * x_hat + self.beta
    self.x_hat = x_hat

    return out
  
  def backward(self, dout, track=False):
    #dout shape: (batch, channels, out_h, out_w)
    batch, channels, out_h, out_w = dout.shape
    N = batch * out_h * out_w

    x_hat = self.x_hat
    var = self.var
    mean = self.mean
    eps = self.eps

    #gradients of gamma and beta
    dgamma = np.sum(dout * x_hat, axis=(0,2,3), keepdims=True)
    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True)

    self.dgamma = dgamma
    self.dbeta = dbeta

    #gradient wrt x
    dx_hat = dout * self.gamma

    dvar = np.sum(dx_hat * (self.x - mean) * -0.5 * (var + eps)**(-3/2), axis=(0,2,3), keepdims=True)
    dmean = np.sum(dx_hat * -1 / np.sqrt(var + eps), axis=(0,2,3), keepdims=True) + dvar * np.sum(-2 * (self.x - mean), axis=(0,2,3), keepdims=True) / N

    dx = dx_hat / np.sqrt(var + eps) + dvar * 2 * (self.x - mean) / N + dmean
    print("mean:", np.mean(self.mean))
    print("var:", np.mean(self.var))
    print("gamma:", np.mean(self.gamma))
    print("beta:", np.mean(self.beta))
    print("xhat max:", np.max(x_hat))
    print("dout max:", np.max(dout))

    return dx
  
  def set_params(self, params):
    self.gamma, self.beta = params
    
  def get_params(self):
    return [self.gamma, self.beta]
  
  def get_grads(self):
    return [self.dgamma, self.dbeta]
  
  def get_metrics(self):
    return {}

class MaxPool2D(Layer):
  is_visualizable = True

  def __init__(self, size, stride):
    self.size = size
    self.stride = stride if stride is not None else size
    self.activation = None

  def forward(self, x, track=False):
    #x shape: (batch, channels, H, W)
    self.x = x
    batch, channels, H, W = x.shape
    k = self.size
    s = self.stride

    out_h = (H - k) // s + 1
    out_w = (W - k) // s + 1

    out = np.zeros((batch, channels, out_h, out_w))

    #store mask for backward pass
    self.mask = np.zeros_like(x)

    for n in range(batch):
      for c in range(channels):
        for i in range(out_h):
          for j in range(out_w):
            h_start = i * s
            w_start = j * s
            region = x[n, c, h_start:h_start+k, w_start:w_start+k]
            max_val = np.max(region)
            out[n, c, i, j] = max_val

            #create mask for backward pass
            max_mask = (region == max_val)
            self.mask[n, c, i*s:(i+1)*s, j*s:(j+1)*s] = max_mask
    
    if track:
      self.activation = out.copy()
      self.last_out = out
    else:
      self.activation = None

    return out
  
  def backward(self, dout, track=False):
    #dout shape: (batch, channels, out_h, out_w)
    batch, channels, out_h, out_w = dout.shape
    k = self.size
    s = self.stride

    dx = np.zeros_like(self.x)

    for n in range(batch):
      for c in range(channels):
        for i in range(out_h):
          for j in range(out_w):
            h_start = i * s
            w_start = j * s

            #gradient to distribute
            grad = dout[n, c, i, j]

            #mask region
            region_mask = self.mask[n, c, h_start:h_start+k, w_start:w_start+k]

            #send gradient only to max location
            dx[n, c, h_start:h_start+k, w_start:w_start+k] += region_mask * grad

    if track:
      self.last_dout = dout

    return dx
  
  def get_activation(self):
    return self.activation
  
  def set_params(self, params):
    pass
    
  def get_params(self):
    return []
  
  def get_grads(self):
    return []
  
  def get_metrics(self):
    return {}

class Flatten(Layer):
  def forward(self, x, track=False):
    self.original_shape = x.shape
    out = x.reshape(x.shape[0], -1)

    if track:
      self.last_out = out

    return out
  
  def backward(self, dout, track=False):
    if track:
      self.last_dout = dout

    return dout.reshape(self.original_shape)
  
  def set_params(self, params):
    pass
    
  def get_params(self):
    return []
  
  def get_grads(self):
    return []
  
  def get_metrics(self):
    return {}
