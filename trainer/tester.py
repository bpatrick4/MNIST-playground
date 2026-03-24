class Tester:
  def __init__(self, model, test_sets, epochs, batch_size, test_imgs):
    self.model = model
    self.epochs = epochs
    self.batch_size = batch_size
    self.test_imgs = test_imgs

    self.X_test_raw = test_sets[0]
    self.Y_test_raw = test_sets[1]

    self.X_test_small_raw = test_sets[2]
    self.Y_test_small_raw = test_sets[3]
    self.x_single_raw = test_sets[4]

    self.Y_test_onehot = test_sets[5]
    self.Y_test_small_onehot = test_sets[6]

    self.X_test_cnn = test_sets[7]
    self.X_test_small_cnn = test_sets[8]
    self.x_single_cnn = test_sets[9]

  def run_test(self, X, y_true):
    #validate for accuracy
    return self.model.validate(X, y_true)[1]
  
  def printout(self, batches_per_epoch, train_imgs):
    #test on stored weights
    print(f'epochs: {self.epochs} | '
      f'batches per epoch: {batches_per_epoch} | '
      f'total batches: {batches_per_epoch * self.epochs} | '
      f'images used in training run: {train_imgs} | '
      f'images used in test set: {self.test_imgs}\n'
      'Testing...')
