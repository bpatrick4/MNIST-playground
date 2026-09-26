from utils.dataset_utils import test_sets_mlp, dataset_inputs
from models.models import MLP
from dashboard.dashboard import DashboardTest

#set datasets
X_test_raw, Y_test_raw, x_single_raw, Y_test_onehot = test_sets_mlp
train_imgs, val_imgs, test_imgs, n = dataset_inputs

#set model type
test_model = MLP()
dashboard = DashboardTest(test_model)

#load model weights
test_model.load(f'weights\weights_mlp\e32\_781.25bpe_lr-0.002_sgd_t-50000_v-10000.npz')

#validate for accuracy on stored weights
acc = test_model.validate(X_test_raw, Y_test_onehot) 
print('test set accuracy:', acc[1])

#plot confusion and per class accuracy
dashboard.plot_test_confusion(X_test_raw, Y_test_raw)

#print out features maps of digit {n}
print(f'feature map guess:{test_model.predict(x_single_raw)}-{n}')
dashboard.plot_test_feature_maps(x_single_raw)
