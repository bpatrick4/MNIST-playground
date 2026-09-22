from mnist_data.dataset_utils import test_sets_cnn, dataset_inputs
from models.models import CNN, Tester_cnn
from dashboard.dashboard import DashboardTest

#set datasets
train_imgs, val_imgs, test_imgs, n = dataset_inputs

#set model type
test_model = CNN()

#set tester
tester = Tester_cnn(test_model, test_sets_cnn)
dashboard = DashboardTest(tester)

#load model weights
test_model.load(f'weights\weights_cnn\e12\weights_lr0.004_bpe781.25_adam_BN_train50000_validate10000.npz')

#validate for accuracy on stored weights
acc = tester.run_test(tester.X_test_small_cnn, tester.Y_test_small_onehot)
print('test set accuracy:', acc)

#plot confusion and per class accuracy
dashboard.plot_test_confusion(tester.X_test_small_cnn, tester.Y_test_small_raw)

#print out features maps of digit {n}
print(f'feature map guess:{tester.model.predict(tester.x_single_cnn)}-{n}')
dashboard.plot_test_feature_maps(tester.x_single_cnn)
