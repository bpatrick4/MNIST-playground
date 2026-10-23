import numpy as np

class Optimizer:
  def step(self, layers):
    raise NotImplementedError

class SGD(Optimizer):
  def __init__(self, lr):
    self.lr = lr
    self.type = "sgd"

  def step(self, layers):
    for layer in layers:
      for p, g in zip(layer.get_params(), layer.get_grads()):
          p -= self.lr * g


class Adam(Optimizer):
  def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.t = 0
    self.type = "adam"

  def step(self, layers):
    self.t += 1
    for layer in layers:
      params = layer.get_params()
      grads = layer.get_grads()

      if not hasattr(layer, "m"):
        layer.m = [np.zeros_like(p) for p in params]
        layer.v = [np.zeros_like(p) for p in params]

      for i, (p, g) in enumerate(zip(params, grads)):
        layer.m[i] = self.beta1 * layer.m[i] + (1 - self.beta1) * g
        layer.v[i] = self.beta2 * layer.v[i] + (1 - self.beta2) * (g * g)

        m_hat = layer.m[i] / (1 - self.beta1 ** self.t)
        v_hat = layer.v[i] / (1 - self.beta2 ** self.t)

        p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)