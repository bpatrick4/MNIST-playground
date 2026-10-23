from utils.dataset_utils import test_sets_cnn, dataset_inputs
from models.models import CNN
from dashboard.dashboard import DashboardTest

#set datasets
X_test_small_cnn, Y_test_small_raw, x_single_cnn, Y_test_small_onehot = test_sets_cnn
train_imgs, val_imgs, test_imgs, n = dataset_inputs

#set model type
test_model = CNN()
dashboard = DashboardTest(test_model)

#load model weights
test_model.load(f'weights\weights_cnn\e8\_844.0bpe_lr-0.003_sgd_t-54016_v-5984.npz')

#validate for accuracy on stored weights
acc = test_model.validate(X_test_small_cnn, Y_test_small_onehot)
print('test set accuracy:', acc[1])

#plot confusion and per class accuracy
dashboard.plot_test_confusion(X_test_small_cnn, Y_test_small_raw)

#print out features maps of digit {n}
print(f'feature map guess:{test_model.predict(x_single_cnn)}-{n}')
dashboard.plot_test_feature_maps(x_single_cnn)
