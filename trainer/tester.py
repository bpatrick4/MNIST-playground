class Tester:
  def __init__(self, model, test_sets):
    self.model = model

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
