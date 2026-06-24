from mnist_data.dataset_utils import test_sets, dataset_inputs
from models.models import CNN, MLP
from trainer.tester import Tester
from dashboard.dashboard import DashboardTest

#set datasets
train_imgs, val_imgs, test_imgs, n = dataset_inputs

#set model type
cnn_model, mlp_model = CNN(), MLP()
test_model = cnn_model

#set tester
tester = Tester(test_model, test_sets)
dashboard = DashboardTest(tester)

#load model weights
#test_model.load(f'weights\weights_cnn\e8\weights_bpe256.0_sgd_train16384_validate2048.npz')

test_model.load(f'weights\weights_cnn_ft\e4\weights_bpe4.0_sgd_train256_validate256.npz')

#test on stored weights
if test_model.type == 'cnn':
  #validate for accuracy
  acc = tester.run_test(tester.X_test_small_cnn, tester.Y_test_small_onehot)
  print('test set accuracy:', acc)
  
  #plot confusion and per class accuracy
  dashboard.plot_test_confusion(tester.X_test_small_cnn, tester.Y_test_small_raw)

  #print out features maps of digit {n}
  print(f'feature map guess:{tester.model.predict(tester.x_single_cnn)}-{n}')
  dashboard.plot_test_feature_maps(tester.x_single_cnn)
elif test_model.type == 'mlp':
  #validate for accuracy
  acc = tester.run_test(tester.X_test_raw, tester.Y_test_onehot)
  print('test set accuracy:', acc)

  #plot confusion and per class accuracy
  dashboard.plot_test_confusion(tester.X_test_raw, tester.Y_test_raw)

  #print out features maps of digit {n}
  print(f'feature map guess:{tester.model.predict(tester.x_single_raw)}-{n}')
  dashboard.plot_test_feature_maps(tester.x_single_raw)
