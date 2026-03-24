from mnist_data.dataset_utils import test_sets, dataset_inputs
from models.models import CNN, MLP
from trainer.tester import Tester
from dashboard.dashboard import DashboardTest

#set variables
train_imgs, val_imgs, test_imgs, n = dataset_inputs
cnn_model, mlp_model = CNN(), MLP()
epochs = 2 ** 1
batch_size = 2 ** 6
batches_per_epoch = (train_imgs / batch_size)

#set model type
test_model = cnn_model

#set tester
tester = Tester(test_model, test_sets, epochs, batch_size, test_imgs)
dashboard = DashboardTest(tester)

#load model weights
#test_model.load(f'weights\weights_{test_model.type}\e{epochs}bpe{batches_per_epoch}traimgs{train_imgs}valimgs{val_imgs}_weights.npz')
test_model.load(f'weights\weights_cnn\e12bpe7.5traimgs480valimgs120_weights_adam.npz')
#test_model.load(f'weights\weights_cnn\e8bpe781.25traimgs50000valimgs10000_weights.npz')

#test on stored weights
tester.printout(batches_per_epoch, train_imgs)

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
