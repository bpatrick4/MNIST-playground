from utils.dataset_utils import train_sets_mlp, dataset_inputs
from models.models import MLP
from dashboard.dashboard import DashboardTrain

X_train, X_val_raw, Y_train_onehot, Y_val_onehot = train_sets_mlp 
train_imgs, val_imgs, _, _ = dataset_inputs
tune_model = MLP()  # training model

#finetuning inputs
epochs = 2 ** 2         # range(2, 4)
batch_size = 2 ** 6     # range(5, 7)
learning_rate = 0.001    # range(0.01, 0.1)
drop_prob = 0.240       # range(0.15, 0.35)
l2_lambda = 0.002       # range(0.001, 0.005)

#optimizer
optimizer='sgd'
beta1=0.9
beta2=0.999
eps=1e-8

#outputs
batches_per_epoch = (train_imgs / batch_size)

#load trained weights
tune_model.load(f'weights\weights_mlp\e8\_15.625bpe_lr-0.001_adam_t-1000_v-1000.npz')

#train model
tune_model.printout(epochs, batches_per_epoch, train_imgs, val_imgs)

tune_model.train_validate(X_train, Y_train_onehot, X_val_raw, Y_val_onehot,
                    epochs=epochs, optimizer=optimizer, lr=learning_rate, batch_size=batch_size, l2_lambda=l2_lambda,
                    beta1=beta1, beta2=beta2, eps=eps)

#save model weights
tune_model.save(f'weights\weights_{tune_model.type}_ft'
           f'\e{epochs}\_{batches_per_epoch}bpe'
           f'_lr-{learning_rate}_{optimizer}'
           f'_t-{train_imgs}_v-{val_imgs}.npz')
#graphs
dashboard = DashboardTrain(tune_model)
dashboard.plot_training_curves()
dashboard.plot_all_overlays()