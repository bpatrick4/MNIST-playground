from mnist_data.dataset_utils import test_sets_mlp, dataset_inputs
from models.models import MLP, Tester_mlp
from dashboard.dashboard import DashboardTest

#set datasets
train_imgs, val_imgs, test_imgs, n = dataset_inputs

#set model type
test_model = MLP()

#set tester
tester = Tester_mlp(test_model, test_sets_mlp)
dashboard = DashboardTest(tester)

#load model weights
test_model.load(f'weights/weights_mlp/e4/weights_lr0.001_bpe187.5_adam_BN_train12000_validate6000.npz')

#validate for accuracy on stored weights
acc = tester.run_test(tester.X_test_raw, tester.Y_test_onehot) 
print('test set accuracy:', acc)

#plot confusion and per class accuracy
dashboard.plot_test_confusion(tester.X_test_raw, tester.Y_test_raw)

#print out features maps of digit {n}
print(f'feature map guess:{tester.model.predict(tester.x_single_raw)}-{n}')
dashboard.plot_test_feature_maps(tester.x_single_raw)
